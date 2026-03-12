import json
import datetime
import contextlib
import arviz as az
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import KFold


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('..')
from src.util import DualLogger, savefig_base
from src.popsynth import PMZLinearInterpolator, PMZWindPileupModel

plt.style.use('./plotstyle.mplstyle')

# Utilities

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into a tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# Splitters
# These classes split a dataset (core_props_df) into training and validation sets

class CHEBoundarySplitter:
    """
    A custom cross-validation splitter designed for 3D stellar grids (M, P, Z) 
    that feature a bounded "cone" of valid parameter space (like CHE).
    
    It guarantees that the geometric boundaries of the parameter space are ALWAYS 
    kept in the training set, preventing 3D interpolators from failing due to 
    extrapolation/convex hull errors. Only the safe, internal points are rotated 
    into the validation sets.
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def extract_boundaries(self, df, m_col='m_zams', p_col='p_spin_zams', z_col='z'):
        """
        Identifies the indices of all points that define the outer envelope 
        of the valid parameter space.
        """
        boundary_idx = set()
        
        for z_val, z_group in df.groupby(z_col):
            # 1. The absolute Mass extremes for this Z (including the cusp)
            boundary_idx.update(z_group[z_group[m_col] == z_group[m_col].min()].index)
            boundary_idx.update(z_group[z_group[m_col] == z_group[m_col].max()].index)
            
            # 2. The Period boundaries for every single mass slice
            for m_val, m_group in z_group.groupby(m_col):
                boundary_idx.update(m_group[m_group[p_col] == m_group[p_col].min()].index)
                boundary_idx.update(m_group[m_group[p_col] == m_group[p_col].max()].index)
                
        return np.array(list(boundary_idx))

    def split(self, df, m_col='m_zams', p_col='p_spin_zams', z_col='z'):
        """
        Yields (train_indices, test_indices) for each fold.
        """
        # 1. Identify all structural boundary points
        boundary_idx = self.extract_boundaries(df, m_col, p_col, z_col)
        
        # 2. Identify the "interior" points that are safe to use for validation
        all_idx = df.index.to_numpy()
        interior_idx = np.setdiff1d(all_idx, boundary_idx)
        
        # 3. Perform K-Fold splits ONLY on the interior points
        for interior_train_idx, interior_test_idx in self.kf.split(interior_idx):
            # The training set is the boundaries PLUS the interior training points
            train_idx = np.concatenate([boundary_idx, interior_idx[interior_train_idx]])
            # The test set is exclusively drawn from the interior
            test_idx = interior_idx[interior_test_idx]
            
            yield train_idx, test_idx

class DecimatedCHEBoundarySplitter:
    """
    Creates a train/test split that selects a genuinely coarser structured grid
    by applying a stride to the unique sorted coordinates of the parameter space.
    
    This performs boundary decimation: systematically lowering the grid resolution 
    while ensuring the absolute maximum boundaries are retained to perfectly 
    preserve the parameter domain volume (the convex hull).
    """
    def __init__(self, strides_dict):
        """
        Parameters:
        -----------
        strides_dict : dict
            Dictionary mapping column names to integer strides 
            (e.g., {'m_zams': 2, 'p_spin_zams': 2, 'z': 3}).
        """
        self.strides_dict = strides_dict

    def get_mask(self, df):
        """
        Returns a boolean pandas Series selecting the decimated coarse grid.
        """
        valid_mask = pd.Series(True, index=df.index)
        
        for col, stride in self.strides_dict.items():
            unique_vals = np.sort(df[col].dropna().unique())
            coarse_vals = list(unique_vals[::stride])
            
            # Ensure the absolute boundaries are retained to preserve domain volume
            if unique_vals[-1] not in coarse_vals:
                coarse_vals.append(unique_vals[-1])
                
            valid_mask &= df[col].isin(coarse_vals)
            
        return valid_mask

    def split(self, df):
        """
        Yields a single (train_idx, test_idx) tuple for the decimated grid.
        This maintains the API pattern of standard cross-validation splitters.
        """
        mask = self.get_mask(df)
        
        # Convert boolean mask to integer positional indices
        train_idx = np.where(mask)[0]
        test_idx = np.where(~mask)[0]
        
        yield train_idx, test_idx

# K-fold tests

def evaluate_fold(fold, train_idx, val_idx, boundary_idx, work_df, frac):
    """
    Evaluates a single fold for both the Physical Model and the Interpolator,
    at a specific sparsity fraction.
    """
    # 1. Simulate Sparsity (while keeping boundaries safe)
    interior_train_idx = np.setdiff1d(train_idx, boundary_idx)
    
    if frac < 1.0:
        # Randomly drop (1-frac) of the interior points
        np.random.seed(fold) # Ensure reproducibility per fold
        keep_size = int(len(interior_train_idx) * frac)
        interior_train_idx = np.random.choice(interior_train_idx, keep_size, replace=False)
        
    # Recombine boundaries with the sparse interior to make the training set
    sparse_train_idx = np.concatenate([boundary_idx, interior_train_idx])
    
    train_df = work_df.loc[sparse_train_idx].copy()
    val_df = work_df.loc[val_idx].copy()
    actual_mf = val_df['m_f'].values

    # ==========================================
    # 2. Evaluate Physical Model
    # ==========================================
    from src.popsynth import PMZWindPileupModel, PMZLinearInterpolator
    phys_model = PMZWindPileupModel(train_df, var='m_f', chain_n=4) 
    phys_model.fit(verbose=False) 
    
    phys_pred, _ = phys_model.get_mf_logtd(
        val_df['m_zams'].values, 
        val_df['p_spin_zams'].values, 
        val_df['z'].values
    )
    phys_rmse = np.sqrt(np.mean((phys_pred - actual_mf)**2))
    
    # ==========================================
    # 3. Evaluate Linear Interpolator
    # ==========================================
    interp_model = PMZLinearInterpolator(train_df, var='m_f', bounds_error=False, fill_value=np.nan)
    
    # Safe row-by-row evaluation for nested 1D interpolators
    interp_pred = []
    for _, row in val_df.iterrows():
        try:
            val = interp_model.get_var(row['m_zams'], row['p_spin_zams'], row['z'])
            interp_pred.append(float(val))
        except Exception:
            interp_pred.append(np.nan)
            
    interp_pred = np.array(interp_pred)
    
    valid_mask = ~np.isnan(interp_pred)
    nan_count = np.sum(~valid_mask)
    nan_frac = nan_count / len(interp_pred)
    
    if np.sum(valid_mask) > 0:
        interp_rmse = np.sqrt(np.mean((interp_pred[valid_mask] - actual_mf[valid_mask])**2))
    else:
        interp_rmse = np.nan 
        
    return {
        'fold': fold,
        'train_frac': frac,
        'train_size': len(train_df),
        'test_size': len(val_df),
        'phys_rmse': phys_rmse,
        'interp_rmse': interp_rmse,
        'interp_nan_frac': nan_frac
    }

# random scarcity test

def evaluate_random_sparse_test(work_df, train_fracs=None, out_dir=None, n_splits=5, n_jobs=-1):
    """
    Executes Experiment 1: Random K-Fold Sparsity Benchmark.
    Randomly drops a percentage of interior grid points and evaluates interpolation.
    Resumes from existing cached results if available.
    """
    if train_fracs is None:
        train_fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    results = []
    fracs_to_run = train_fracs
    res_file = None
    summary_file = None

    if out_dir is not None:
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        res_file = out_dir / "random_sparse_results.csv"
        summary_file = out_dir / "random_sparse_summary.csv"
        
        if res_file.exists():
            print(f"\n[+] Found existing Experiment 1 results at {res_file}.")
            existing_df = pd.read_csv(res_file)
            
            # Convert existing dataframe back to list of dicts to append easily
            results = existing_df.to_dict('records') 
            
            # Using rounding/tolerance could be useful for floats, but direct string matching or 
            # simple set intersection usually handles standard fraction lists gracefully.
            existing_fracs = set(existing_df['train_frac'].tolist())
            fracs_to_run = [f for f in train_fracs if f not in existing_fracs]
            
            if not fracs_to_run:
                print("\n[+] All sparsity fractions already computed. Skipping compute...")
                existing_summary = pd.read_csv(summary_file) if summary_file.exists() else None
                if existing_summary is not None:
                    print("\n=== RANDOM SPARSITY BENCHMARK SUMMARY (CACHED) ===")
                    print(existing_summary.round(3).to_string(index=False))
                return existing_df, existing_summary
            else:
                print(f"[+] Found {len(fracs_to_run)} missing fractions to compute.")

    print("\n" + "="*50)
    print(" EXPERIMENT 1: RANDOM K-FOLD SPARSITY BENCHMARK")
    print("="*50)

    # Extract the boundary points once for the entire grid
    splitter = CHEBoundarySplitter(n_splits=n_splits, random_state=42)
    boundary_idx = splitter.extract_boundaries(work_df)
    
    for frac in tqdm(fracs_to_run, desc="Evaluating Sparsity Fractions"):
        
        # Run folds in parallel to speed up execution
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(fold_i, train_idx, test_idx, boundary_idx, work_df, frac)
            for fold_i, (train_idx, test_idx) in enumerate(splitter.split(work_df))
        )
        
        results.extend(fold_results)
        
        # Incrementally save results after each full fraction finishes
        if res_file is not None:
            pd.DataFrame(results).to_csv(res_file, index=False)

    results_df = pd.DataFrame(results)
    
    # Recalculate summary metrics across all populated results (cached + newly run)
    summary_df = results_df.groupby('train_frac').agg({
        'phys_rmse': ['mean', 'std'],
        'interp_rmse': ['mean', 'std'],
        'interp_nan_frac': ['mean', 'std']
    }).reset_index()
    
    # Flatten MultiIndex columns to make the final CSV clean
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    # Save the updated summary
    if summary_file is not None:
        summary_df.to_csv(summary_file, index=False)
        
    print("\n=== RANDOM SPARSITY BENCHMARK SUMMARY ===")
    print(summary_df.round(3).to_string(index=False))
    
    return results_df, summary_df

# convex hull simplification test

def get_coarse_grid_mask(df, strides_dict):
    """
    Creates a boolean mask that selects a genuinely coarser structured grid
    by applying a stride to the unique sorted coordinates of the parameter space.
    """
    valid_mask = pd.Series(True, index=df.index)
    for col, stride in strides_dict.items():
        unique_vals = np.sort(df[col].dropna().unique())
        coarse_vals = list(unique_vals[::stride])
        # Ensure the absolute boundaries are retained to preserve the parameter domain volume
        if unique_vals[-1] not in coarse_vals:
            coarse_vals.append(unique_vals[-1])
        valid_mask &= df[col].isin(coarse_vals)
    return valid_mask


def evaluate_structured_coarsening_test(work_df, configs=None, out_dir=None):
    """
    Executes Experiment 1B: True Structured Grid Coarsening.
    Reduces the discretization of the grid systematically and tests on the skipped points.
    Resumes from existing cached results if available.
    """
    if configs is None:
        configs = [
            {'name': 'Stride 1', 'm_zams': 1, 'p_spin_zams': 1, 'z': 1},
            {'name': 'Stride 2 (P)', 'm_zams': 1, 'p_spin_zams': 2, 'z': 1},
            {'name': 'Stride 2 (P, M)', 'm_zams': 2, 'p_spin_zams': 2, 'z': 1},
            {'name': 'Stride 2 (All)', 'm_zams': 2, 'p_spin_zams': 2, 'z': 2},
            {'name': 'Stride 3 (All)', 'm_zams': 3, 'p_spin_zams': 3, 'z': 3},
            {'name': 'Stride 4 (All)', 'm_zams': 4, 'p_spin_zams': 4, 'z': 4},
            {'name': 'Stride 5 (All)', 'm_zams': 5, 'p_spin_zams': 5, 'z': 5},
        ]

    results = []
    configs_to_run = configs
    res_file = None

    if out_dir is not None:
        # Ensure out_dir is a Path object
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        res_file = out_dir / "coarsening_results.csv"
        
        if res_file.exists():
            print(f"\n[+] Found existing Experiment 1B results at {res_file}.")
            existing_df = pd.read_csv(res_file)
            
            # Convert existing dataframe back to list of dicts to maintain order and append easily
            results = existing_df.to_dict('records') 
            existing_names = set(existing_df['stride_name'].tolist())
            
            configs_to_run = [c for c in configs if c['name'] not in existing_names]
            
            if not configs_to_run:
                print("\n[+] All configurations already computed. Skipping compute...")
                print("\n=== COARSENING BENCHMARK SUMMARY (CACHED) ===")
                print(existing_df.round(3).to_string(index=False))
                return existing_df, existing_df
            else:
                print(f"[+] Found {len(configs_to_run)} missing configurations to compute.")

    print("\n" + "="*50)
    print(" EXPERIMENT 1B: STRUCTURED GRID COARSENING")
    print("="*50)

    for config in configs_to_run:
        c_name = config['name']
        print(f"\n--- Evaluating at {c_name} ---")
        
        strides_dict = {'m_zams': config['m_zams'], 'p_spin_zams': config['p_spin_zams'], 'z': config['z']}
        
        # Instantiate the splitter with the specific decimation strides
        splitter = DecimatedCHEBoundarySplitter(strides_dict=strides_dict)
        coarse_mask = splitter.get_mask(work_df)
        
        train_df = work_df[coarse_mask].copy()
        val_df = work_df[~coarse_mask].copy()
        
        if len(val_df) == 0:
            val_df = train_df.copy()
            print(f"  ({c_name}: Evaluating baseline on the full dense grid itself)")
        else:
            print(f"  Training points: {len(train_df)} | Holdout points: {len(val_df)}")
            
        if len(train_df) == 0:
            print("  Not enough points to train. Skipping.")
            continue
            
        actual_mf = val_df['m_f'].values

        # ==========================================
        # Evaluate Physical Model
        # ==========================================
        # Assuming PMZWindPileupModel is imported correctly in your main script
        from src.popsynth import PMZWindPileupModel, PMZLinearInterpolator
        phys_model = PMZWindPileupModel(train_df, var='m_f', chain_n=4) 
        phys_model.fit(verbose=False)
        
        phys_pred, _ = phys_model.get_mf_logtd(
            val_df['m_zams'].values, 
            val_df['p_spin_zams'].values, 
            val_df['z'].values
        )
        phys_rmse = np.sqrt(np.mean((phys_pred - actual_mf)**2))
        
        # ==========================================
        # Evaluate Linear Interpolator
        # ==========================================
        interp_model = PMZLinearInterpolator(train_df, var='m_f', bounds_error=False, fill_value=np.nan)
        
        interp_pred = []
        for _, row in val_df.iterrows():
            try:
                val = interp_model.get_var(row['m_zams'], row['p_spin_zams'], row['z'])
                interp_pred.append(float(val))
            except Exception:
                interp_pred.append(np.nan)
                
        interp_pred = np.array(interp_pred)
        valid_mask = ~np.isnan(interp_pred)
        nan_count = np.sum(~valid_mask)
        nan_frac = nan_count / len(interp_pred)
        
        if np.sum(valid_mask) > 0:
            interp_rmse = np.sqrt(np.mean((interp_pred[valid_mask] - actual_mf[valid_mask])**2))
        else:
            interp_rmse = np.nan
            
        print(f"  Physical Model RMSE : {phys_rmse:.3f} M_sun")
        print(f"  Interpolator RMSE   : {interp_rmse:.3f} M_sun")
        print(f"  Interpolator Fails  : {nan_frac*100:.1f}% (Returned NaN)")

        results.append({
            'stride_name': c_name,
            'train_size': len(train_df),
            'test_size': len(val_df),
            'phys_rmse': phys_rmse,
            'interp_rmse': interp_rmse,
            'interp_nan_frac': nan_frac
        })

        # Incrementally save results after each configuration
        if res_file is not None:
            pd.DataFrame(results).to_csv(res_file, index=False)

    results_df = pd.DataFrame(results)
    print("\n=== COARSENING BENCHMARK SUMMARY ===")
    print(results_df.round(3).to_string(index=False))
    
    return results_df, results_df


# dense x sparse grid intersection test
def evaluate_structured_sparse_test(work_df, out_dir=None, force_model_storage=False):
    """
    Tests actual model performance by training ONLY on the intersecting nodes 
    of the real sparse grid, and predicting the rest of the dense parameter space.
    """
    if out_dir is not None:
        met_file = out_dir / "structured_test_metrics.json"
        if met_file.exists() and not force_model_storage:
            print("\n[+] Found cached Experiment 2 metrics. Skipping compute...")
            with open(met_file, "r") as f:
                metrics = json.load(f)
            print("\n=== STRUCTURED GRID TEST RESULTS (CACHED) ===")
            print(f"  Physical Model RMSE : {metrics['phys_rmse']:.3f} M_sun")
            i_rmse = metrics['interp_rmse']
            print(f"  Interpolator RMSE   : {i_rmse:.3f} M_sun" if i_rmse is not None else "  Interpolator RMSE   : NaN M_sun")
            print(f"  Interpolator Fails  : {metrics['interp_nan_frac']*100:.1f}% (Returned NaN)")
            
            # Return None for models since we skipped training, but check if interp_model exists
            phys_model = None
            interp_model = None
            if (out_dir / "interp_model.pkl").exists():
                try: interp_model = joblib.load(out_dir / "interp_model.pkl")
                except: pass
                
            return phys_model, interp_model, metrics

    print("\n" + "="*50)
    print(" EXPERIMENT 2: STRUCTURED SPARSE GRID GENERALIZATION")
    print("="*50)
    
    # 1. Define the overlapping sparse grid nodes
    Z_sparse_div = [0.02, 0.10, 0.40]
    M_sparse = [20.0, 30.0, 40.0, 90.0, 100.0, 200.0, 300.0]
    P_sparse = [0.40, 0.50, 1.00, 1.50, 2.00] # Intersecting periods

    tol = 1e-4
    
    # Handle absolute vs relative metallicity gracefully (Z_sun = 0.014)
    z_is_div = work_df['z'].max() >= 0.02 
    z_targets = Z_sparse_div if z_is_div else [z * 0.014 for z in Z_sparse_div]

    mask_z = work_df['z'].apply(lambda x: any(abs(x - z_val) < tol for z_val in z_targets))
    mask_m = work_df['m_zams'].apply(lambda x: any(abs(x - m_val) < tol for m_val in M_sparse))
    mask_p = work_df['p_spin_zams'].apply(lambda x: any(abs(x - p_val) < tol for p_val in P_sparse))
    
    sparse_mask = mask_z & mask_m & mask_p
    
    train_df = work_df[sparse_mask].copy()
    val_df = work_df[~sparse_mask].copy()
    
    print(f"Mock Sparse Grid (Train): {len(train_df)} valid CHE points")
    print(f"Dense Grid Holdout (Test): {len(val_df)} valid CHE points")
    
    if len(train_df) == 0:
        print("Error: Could not match any points to the sparse grid definitions!")
        return

    # ==========================================
    # Evaluate Physical Model
    # ==========================================
    print("\nFitting Physical Model on Sparse Grid (takes a moment)...")
    phys_model = PMZWindPileupModel(train_df, var='m_f', chain_n=4) 
    phys_model.fit(verbose=False)
    phys_pred, _ = phys_model.get_mf_logtd(
        val_df['m_zams'].values, 
        val_df['p_spin_zams'].values, 
        val_df['z'].values
    )
    actual_mf = val_df['m_f'].values
    phys_rmse = np.sqrt(np.mean((phys_pred - actual_mf)**2))
    
    # ==========================================
    # Evaluate Interpolator
    # ==========================================
    print("Fitting Interpolator on Sparse Grid...")
    interp_model = PMZLinearInterpolator(train_df, var='m_f', bounds_error=False, fill_value=np.nan)
    interp_pred = []
    for _, row in val_df.iterrows():
        try:
            val = interp_model.get_var(row['m_zams'], row['p_spin_zams'], row['z'])
            interp_pred.append(float(val))
        except Exception:
            interp_pred.append(np.nan)
            
    interp_pred = np.array(interp_pred)
    valid_mask = ~np.isnan(interp_pred)
    nan_count = np.sum(~valid_mask)
    nan_frac = nan_count / len(interp_pred)
    
    if np.sum(valid_mask) > 0:
        interp_rmse = np.sqrt(np.mean((interp_pred[valid_mask] - actual_mf[valid_mask])**2))
    else:
        interp_rmse = np.nan
        
    print("\n=== STRUCTURED GRID TEST RESULTS ===")
    print(f"  Physical Model RMSE : {phys_rmse:.3f} M_sun")
    print(f"  Interpolator RMSE   : {interp_rmse:.3f} M_sun")
    print(f"  Interpolator Fails  : {nan_frac*100:.1f}% (Returned NaN)")

    # NEW: Package results for saving
    metrics = {
        'phys_rmse': float(phys_rmse),
        'interp_rmse': float(interp_rmse) if not np.isnan(interp_rmse) else None,
        'interp_nan_frac': float(nan_frac)
    }
    
    return phys_model, interp_model, metrics

# Test suite

def save_test_suite_benchmark(out_dir, results_df, summary, coarse_df, coarse_summary, struct_metrics, phys_model, interp_model):
    """
    Central catch-all save function. Writes out the dataframes, JSON metrics, 
    and importantly, uses joblib to pickle the fully trained models to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if results_df is not None:
        results_df.to_csv(out_dir / "random_sparse_results.csv", index=False)
    if summary is not None:
        summary.to_csv(out_dir / "random_sparse_summary.csv", index=False)
        
    if coarse_df is not None:
        coarse_df.to_csv(out_dir / "coarsening_results.csv", index=False)
    if coarse_summary is not None:
        coarse_summary.to_csv(out_dir / "coarsening_summary.csv", index=False)
        
    if struct_metrics is not None:
        with open(out_dir / "structured_test_metrics.json", "w") as f:
            json.dump(struct_metrics, f, indent=4)

    # --- Pickling the Models Centralized Here ---
    if phys_model is not None:
        print(f"\n[+] Pickling Physical Model to {out_dir}...")
        joblib.dump(phys_model, out_dir / "extrap_phys_model.pkl")
        
        if hasattr(phys_model, 'trace') and phys_model.trace is not None:
            az.to_netcdf(phys_model.trace, out_dir / "extrap_phys_model_trace.nc")
            
    if interp_model is not None:
        print(f"[+] Pickling Interpolator Model to {out_dir}...")
        joblib.dump(interp_model, out_dir / "extrap_interp_model.pkl")

def run_test_suite(work_df, output_folder, run_id, structured_configs=None, train_fracs=None, force_model_storage=False):
    original_affinity = None
    try:
        # Hardware Lock to P-Cores
        import os
        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(range(12)))
    except AttributeError:
        pass # Handle Windows/psutil fallback if needed
        
    try:        
        # Ensure 'z_key' exists for the interpolator class
        if 'z_key' not in work_df.columns:
            work_df['z_key'] = work_df['z'].astype(str)

        # -----------------------------------------------------
        # USER-DEFINED RUN ID
        # Change this string to start a completely fresh run!
        # -----------------------------------------------------        
        out_dir = output_folder / f"benchmark_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Dual Logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = out_dir / f"benchmark_{timestamp}.log"
        sys.stdout = DualLogger(log_filepath)
        
        print(f"=== Starting Benchmark Run: {run_id} ===")
        print(f"Output Directory: {out_dir}")
        print(f"Logging Output To: {log_filepath}")

        # --- Run Experiment 1: Random Sparsity ---
        results_df, summary = evaluate_random_sparse_test(work_df, train_fracs=train_fracs, out_dir=out_dir)
        # Intermediate cache save (only if we actually ran the compute and the file isn't there)
        if results_df is not None and not (out_dir / "random_cv_results.csv").exists():
            results_df.to_csv(out_dir / "random_cv_results.csv", index=False)
            summary.to_csv(out_dir / "random_cv_summary.csv")
        
        # --- Run Experiment 1B: Structured Grid Coarsening ---
        if structured_configs is None:
            structured_configs = [
                {'name': 'Stride 1',                'm_zams': 1, 'p_spin_zams': 1, 'z': 1},
                #{'name': 'Stride 2 (P)',            'm_zams': 1, 'p_spin_zams': 2, 'z': 1},
                {'name': 'Stride 2 (P, M)',         'm_zams': 2, 'p_spin_zams': 2, 'z': 1},
            # {'name': 'Stride 3 (P)',            'm_zams': 2, 'p_spin_zams': 3, 'z': 1},
                {'name': 'Stride 3 (P, M)',         'm_zams': 3, 'p_spin_zams': 3, 'z': 1},
            # {'name': 'Stride 4 (P)',            'm_zams': 3, 'p_spin_zams': 4, 'z': 1},
                {'name': 'Stride 4 (P, M)',         'm_zams': 4, 'p_spin_zams': 4, 'z': 1},
                {'name': 'Stride 5 (P, M)',         'm_zams': 5, 'p_spin_zams': 5, 'z': 1},
                {'name': 'Stride 6 (P, M)',         'm_zams': 6, 'p_spin_zams': 6, 'z': 1},
                {'name': 'Stride 6 (P, M) + 2 (Z)',         'm_zams': 6, 'p_spin_zams': 6, 'z': 2},
                {'name': 'Stride 6 (P, M) + 3 (Z)',         'm_zams': 6, 'p_spin_zams': 6, 'z': 3},
            ]
        coarse_df, coarse_summary = evaluate_structured_coarsening_test(work_df, configs=structured_configs, out_dir=out_dir)
        if coarse_df is not None and not (out_dir / "coarsening_results.csv").exists():
            coarse_df.to_csv(out_dir / "coarsening_results.csv", index=False)
            coarse_summary.to_csv(out_dir / "coarsening_summary.csv", index=False)
        
        # --- Run Experiment 2: Structured Extrapolation Test ---
        phys_model, interp_model, struct_metrics = evaluate_structured_sparse_test(work_df, out_dir=out_dir, force_model_storage=force_model_storage)
        if struct_metrics is not None and not (out_dir / "structured_test_metrics.json").exists():
            with open(out_dir / "structured_test_metrics.json", "w") as f:
                json.dump(struct_metrics, f, indent=4)

        # --- Save Everything to Final Folder ---
        # This acts as a safe catch-all to dump the MCMC trace and Pickles
        save_test_suite_benchmark(out_dir, results_df, summary, coarse_df, coarse_summary, struct_metrics, phys_model, interp_model)

    finally:
        # Restore stdout if we hijacked it
        if isinstance(sys.stdout, DualLogger):
            sys.stdout.log_file.close()
            sys.stdout = sys.stdout.terminal
            
        # Hardware Unlock
        if original_affinity is not None:
            import os
            os.sched_setaffinity(0, original_affinity)