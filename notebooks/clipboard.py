def read_system(model_path, z_key):
    """Read data for a single MESA run."""
    
    flag_cols, prof_f, h = get_system_flags(model_path)
    is_crit_at_zams = flag_cols[1]
    is_He_depleted = flag_cols[4]
        
    if is_crit_at_zams or not is_He_depleted:
        # Systems that are critically rotating at ZAMS crash immediately
        # so we have no final data for them.
        # Systems that do not complete He burning are not evolved enough
        # for us to use.
        m_zams = float(model_path.name.split('_')[0].lstrip('m').replace('d', 'e'))
        p_spin_zams = float(model_path.name.split('_')[1].lstrip('p').replace('d', 'e'))
        cols = [np.nan]*CORE_PROPS_FLOAT_COL_N + flag_cols
        cols[CORE_PROPS_HEADER.index('z_key')] = z_key
        cols[CORE_PROPS_HEADER.index('m_zams')] = m_zams
        cols[CORE_PROPS_HEADER.index('p_spin_zams')] = p_spin_zams
        cols[CORE_PROPS_HEADER.index('p_orb_zams')] = p_spin_zams
    else:
        logs = mr.MesaLogDir(str(model_path/'LOGS'))
        prof_model_numbers = logs.model_numbers
        
        zams_i = np.where(h.surf_avg_omega > 0)[0][0]
        wr0_i = np.where(h.surface_he4 > y_0)[0][0]
        wr1_i = np.where(h.surface_he4 > y_0 + delta_y)[0][0]
        tams_i = np.where(h.center_h1 < 1e-6)[0][0]
        for tahems_th in [1e-3, 1e-2, 1e-1]:
            try:
                tahems_i = np.where((h.center_h1 < 1e-3) & (h.center_he4 < tahems_th))[0][0]
            except IndexError:
                continue
            else:
                break
        
        model_number_zams = h.model_number[zams_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_zams)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_zams = logs.profile_data(model_number=nearest_model_number)
        m_zams = float(model_path.name.split('_')[0].lstrip('m').replace('d', 'e'))
        p_spin_zams = float(model_path.name.split('_')[1].lstrip('p').replace('d', 'e'))
        r_zams = prof_zams.radius[0]
        log_j_zams = np.log10(prof_zams.J_inside[0])
        conv_core_bound_i_zams = len(prof_zams.mixing_type) - np.where(prof_zams.mixing_type[::-1] != 1)[0][0] - 1
        m_c_zams = prof_zams.mass[conv_core_bound_i_zams]
        r_c_zams = prof_zams.radius[conv_core_bound_i_zams]
        log_rho_c_zams = prof_zams.logRho[conv_core_bound_i_zams]
        age = prof_zams.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_zams, p_zams, _a_zams, _qzams, _tzams = wi.integrate(age)
        p_orb_zams = p_zams.to(u.d).value
        
        model_number_wr0 = h.model_number[wr0_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_wr0)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_wr0 = logs.profile_data(model_number=nearest_model_number)
        m_wr0 = h.star_mass[wr0_i]
        p_spin_wr0 = 2*np.pi / h.surf_avg_omega[wr0_i] * u.s.to(u.d)
        r_wr0 = prof_wr0.radius[0]
        log_j_wr0 = np.log10(prof_wr0.J_inside[0])
        conv_core_bound_i_wr0 = len(prof_wr0.mixing_type) - np.where(prof_wr0.mixing_type[::-1] != 1)[0][0] - 1
        m_c_wr0 = prof_wr0.mass[conv_core_bound_i_wr0]
        r_c_wr0 = prof_wr0.radius[conv_core_bound_i_wr0]
        log_rho_c_wr0 = prof_wr0.logRho[conv_core_bound_i_wr0]
        age = prof_wr0.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_wr0, p_wr0, _a_wr0, _qwr0, _twr0 = wi.integrate(age)
        p_orb_wr0 = p_wr0.to(u.d).value
        
        model_number_wr1 = h.model_number[wr1_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_wr1)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_wr1 = logs.profile_data(model_number=nearest_model_number)
        m_wr1 = h.star_mass[wr1_i]
        p_spin_wr1 = 2*np.pi / h.surf_avg_omega[wr1_i] * u.s.to(u.d)
        r_wr1 = prof_wr1.radius[0]
        log_j_wr1 = np.log10(prof_wr1.J_inside[0])
        conv_core_bound_i_wr1 = len(prof_wr1.mixing_type) - np.where(prof_wr1.mixing_type[::-1] != 1)[0][0] - 1
        m_c_wr1 = prof_wr1.mass[conv_core_bound_i_wr1]
        r_c_wr1 = prof_wr1.radius[conv_core_bound_i_wr1]
        log_rho_c_wr1 = prof_wr1.logRho[conv_core_bound_i_wr1]
        age = prof_wr1.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_wr1, p_wr1, _a_wr1, _qwr1, _twr1 = wi.integrate(age)
        p_orb_wr1 = p_wr1.to(u.d).value
        
        model_number_tams = h.model_number[tams_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_tams)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_tams = logs.profile_data(model_number=nearest_model_number)
        m_tams = h.star_mass[tams_i]
        p_spin_tams = 2*np.pi / h.surf_avg_omega[tams_i] * u.s.to(u.d)
        r_tams = prof_tams.radius[0]
        log_j_tams = np.log10(prof_tams.J_inside[0])
        conv_core_bound_i_tams = len(prof_tams.mixing_type) - np.where(prof_tams.mixing_type[::-1] != 1)[0][0] - 1
        m_c_tams = prof_tams.mass[conv_core_bound_i_tams]
        r_c_tams = prof_tams.radius[conv_core_bound_i_tams]
        log_rho_c_tams = prof_tams.logRho[conv_core_bound_i_tams]
        age = prof_tams.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_tams, p_tams, _a_tams, _qtams, _ttams = wi.integrate(age)
        p_orb_tams = p_tams.to(u.d).value
        
        model_number_tahems = h.model_number[tahems_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_tahems)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_tahems = logs.profile_data(model_number=nearest_model_number)
        m_tahems = h.star_mass[tahems_i]
        p_spin_tahems = 2*np.pi / h.surf_avg_omega[tahems_i] * u.s.to(u.d)
        r_tahems = prof_tahems.radius[0]
        log_j_tahems = np.log10(prof_tahems.J_inside[0])
        conv_core_bound_i_tahems = len(prof_tahems.mixing_type) - np.where(prof_tahems.mixing_type[::-1] != 1)[0][0] - 1
        m_c_tahems = prof_tahems.mass[conv_core_bound_i_tahems]
        r_c_tahems = prof_tahems.radius[conv_core_bound_i_tahems]
        log_rho_c_tahems = prof_tahems.logRho[conv_core_bound_i_tahems]
        age = prof_tahems.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_tahems, p_tahems, _a_tahems, _qtahems, _ttahems = wi.integrate(age)
        p_orb_tahems = p_tahems.to(u.d).value
        
        m_f = prof_f.mass[0]
        p_spin_f = 2*np.pi / prof_f.omega[-1] * u.s.to(u.d)
        r_f = prof_f.radius[0]
        log_j_f = np.log10(prof_f.J_inside[0])
        conv_core_bound_i_f = len(prof_f.mixing_type) - np.where(prof_f.mixing_type[::-1] != 1)[0][0] - 1
        m_c_f = prof_f.mass[conv_core_bound_i_f]
        r_c_f = prof_f.radius[conv_core_bound_i_f]
        log_rho_c_f = prof_f.logRho[conv_core_bound_i_f]
        age = prof_f.star_age
        wi = WindIntegrator(model_path, q0=1)
        _m_f, p_f, a_f, _qf, _tf = wi.integrate(age)
        p_orb_f = p_f.to(u.d).value        
        
        t_c = coalescence_time(m_f, a_f, q=1)
        t_d = t_c + age
        log_t_d = np.log10(t_d)
        
        core_props_cols =[
            z_key,
            m_zams,
            m_wr0,
            m_wr1,
            m_tams,
            m_tahems,
            m_f,
            m_c_zams,
            m_c_wr0,
            m_c_wr1,
            m_c_tams,
            m_c_tahems,
            m_c_f,
            p_spin_zams,
            p_spin_wr0,
            p_spin_wr1,
            p_spin_tams,
            p_spin_tahems,
            p_spin_f,
            r_zams,
            r_wr0,
            r_wr1,
            r_tams,
            r_tahems,
            r_f,
            r_c_zams,
            r_c_wr0,
            r_c_wr1,
            r_c_tams,
            r_c_tahems,
            r_c_f,
            log_rho_c_zams,
            log_rho_c_wr0,
            log_rho_c_wr1,
            log_rho_c_tams,
            log_rho_c_tahems,
            log_rho_c_f,
            log_j_zams,
            log_j_wr0,
            log_j_wr1,
            log_j_tams,
            log_j_tahems,
            log_j_f,
            log_t_d
        ]        
        cols = core_props_cols + flag_cols
        
    row = pd.DataFrame([cols], columns=CORE_PROPS_HEADER)     
    #print(row.memory_usage(deep=True).sum()/1024**2)   
    return row  