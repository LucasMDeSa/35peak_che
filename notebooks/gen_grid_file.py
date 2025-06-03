import numpy as np

min_m = 3
max_m = 300
size_m = 10

min_p = 0.5
max_p = 4
size_p = 10

masses = np.logspace(np.log10(min_m),
                     np.log10(max_m),
                     size_m)

orbital_periods = np.linspace(min_p,
                              max_p,
                              size_p)

omegas = 2*np.pi/orbital_periods

settings = np.array([masses, omegas]).T

np.savetxt('input_grid', settings)