

input :  input data
parameters : hyperparameters + training parameters
results : output graphs/output images/output data/etc
src : all other Python files

compile : it is for my (Damien) setup, because I am lazzy to change 20 times the setup of my code editor


I  created some branches for better management of the work. Please commit on main only working code.  

# Input

## emulator_1100box_planck_products

Contains the raw halo counts from select emulator boxes of the AbacusCosmos simulation suite (https://lgarrison.github.io/AbacusCosmos/simulations/). AbacusCosmos is used to replace the VELMASS simulations because they are publicly available and have code for generating the 2LPT density field. The emulator boxes are chosen since they have boxes that vary along $\Omega_m$ (and related parameters) specifically, allowing reproduction of the varying cosmology test of Ramanah et al. 2020. The z=0.3 slice is used since it is the lowest redshift slice, while Ramanah et al. 2020 used z=0.

Box 00 has the default Planck cosmology and should be used for training, while Box 03 and 04 have higher and lower $\Omega_m$ respectively, and should be used for testing on a variable cosmology.

*Mike: I will process these raw halo files and upload a grid containing the halo counts used by the critic.*
