# EOSC556B-2026_LN 
## Multichannel Analysis of Surface Waves (MASW) inversion for a synthetic multilayer system 

Run this project in your browser:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lambertnguyen-git/EOSC556B-2026_LN/main?urlpath=%2Fdoc%2Ftree%2FMASW_Sim.ipynb)

## REPOSITORY COMPONENTS
- `MASW_Sim.ipynb`: main notebook for synthetic MASW forward modeling and inversion
- `License`: standard license for use
- `environment.yml`: conda environment (compatible with mybinder.org)
- `project_utils.py`: will contain the functions at code cleanup (in progress since code is not finished)
- `test_project_util.py`: unit tests for helper functions (in progress)

> This is a work in progress. As such, functions have not been pulled out of the main code for project_utils yet. 

## BACKGROUND 
Multichannel Analysis of Surface Waves (MASW) is a geophysical method used to estimate shear wave velocity (Vs) profiles. Shear wave velocity is a proxy for soil stiffness and is widely used in geotechnical and earthquake engineering for site classification and assessment of seismic hazards such as liquefaction. MASW involves deploying an array of geophones at fixed intervals, generating surface waves (Rayleigh waves), recording wavefields and extracting dispersion curves. The dispersion curves are then inverted to estimate subsurface shear wave velocity profiles. 

While direct measurements of shear wave velocity (e.g., downhole testing) provide higher accuracy, MASW offers a non-invasive and efficient alternative for characterizing larger areas. This efficiency comes at the cost of increased uncertainty due to the inversion process. 

## QUESTION 
How can shear-wave velocity profiles be recovered from Rayleigh wave dispersion using nonlinear inversion, and how sensitive are the solutions to subsurface layering and velocity contrasts?

## APPROACH 
0. Initial Vs, Vp, thickness, and densities were set. 
1. The model will be setup for a 2 layer system with thicknesses 5 m and 10 m. A half step is included. 
2. Simpeg does not have a forward operator for MASW. There are several MASW forward operators that are avaialble for python. Two of hese include disba and pysurf96. disba was used for this initial run This is included within the environment file. 2a. Later, we will try pysurf96 as the forward operator to understand if there are differences with disba. disba: https://github.com/keurfonluu/disba 
3. The model (and disba) is dependent on layer thicknesses, Vs, p-wave velocity (Vp), and density (rho).    
- 3a. We will keep the layer thicknesses and number of layers constant for this code swap. We will potentially vary these at a later stage. 
- 3b. We are working to solve Vs in this inversion. 
- 3c. Vp is typically 2x faster than Vs. We will keep this simplifying assumption to reduce the variables. 
- 3d. Density will also be kept constant for this code swap. The change in the density may lead to changes in the dispersion curves but this expected to be a low-sensitivity parameter. This may be adjusted at a later stage, but is a low priority. 
4. We will apply the steepest descent gradient method to converge to the solution. Newton's method may be checked prior to the final presentation.

## USAGE 
1. Clone the repository: 
```bash
git clone https://github.com/lambertnguyen-git/EOSC556B-2026_LN
cd EOSC556B-2026_LN 
```
2. Create and activate the environment:
```bash 
conda env create -f environment.yml 
conda activate eosc-556B
```
> WINDOWS USERS: you may have to adjust the environment file from python-mumps to -pydiso
3. Run the main notebook: Open: MASW_Sim.ipynb 
Run all cells from top to bottom 
4. Expected output: 
Synthetic dispersion curve with noise 
Inverted Vs curves 
Observed vs Predicted Dispersion Curves 
> This project is in progress so these plots may not be ready:
Convergence Figure (Objective function) 

## NEXT STEPS
> These are some ideas for next steps. Only some of them will be implemented
- Check noise sensitivity
- Add more soil layers
- Check sensitivities for density, Vp
- Try Newton method
- Check pysurf96 (another forward operator) differences

## ARTIFICIAL INTELLIGENCE USE
AI tools, including the built-in AI assistant for VS Code, were used to troubleshoot coding issues.
