# cell_wall_simulations
Compilation of various draft scripts for modeling the Gram-positive cell wall.

1) wall_growth_kit.py is a library of functions that are used for the simulations of cell wall growth. Within these, the following are particularly relevant:
sim_growth_biomass_strain_stiffening
This is for the biomass-pressure model with strain-stiffening. Note, we don’t actually observe consistency with theory for this simulation set yet.

sim_growth_abrupt_strain_stiffening
This is for the strain-stiffening model with fixed pressure. This gives results that agree with theory.

sim_growth_abrupt_v2
This is for the regular, all-in-one hydrolysis model with fixed pressure and no strain stiffening. This gives results that agree with theory.

2) Jupyter notebooks:

growth_rate_radius_solver_v3_strain_simulations_individual_case.ipynb was used to generate dynamical figures for cell radius, length, stress and strain (both in the fingertip regime and out of it) for the strain-stiffening case.

growth_rate_radius_solver_v3.ipynb was used to generate phase space figures for cell radius and growth rate in the strain-stiffening model from theory predictions (as outlined in the model, note that I tried a couple of ways but these are described in the notebook).

growth_rate_radius_solver_v2.ipynb was used to generate phase space figures from theory, and theory-simulation comparisons, for the non-strain-stiffening model.

230130_abrupt_hydrolysis_phase_space_model.ipynb was used to generate phase space figures from simulations for the non-strain stiffening model. However, this was very preliminary and just showed color based on “radius increases from starting point of 1” vs. “radius decreases from starting point of 1” for different anisotropies and poisson ratios.
