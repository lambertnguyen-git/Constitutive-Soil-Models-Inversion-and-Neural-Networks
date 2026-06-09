# UBC EOSC '26 Capstone Project
## Neural Network-Assisted Determination and Inversion of Constitutive Model Parameters

>WORK IN PROGRESS - README NOT READY

Run this project in your browser:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lambertnguyen-git/Constitutive-Soil-Models-Inversion-and-Neural-Networks/HEAD?urlpath=%2Fdoc%2Ftree%2FInversion_MC.ipynb)


## REPOSITORY COMPONENTS




## BACKGROUND AND APPROACH
Calibrating constitutive soil model parameters from laboratory triaxial test data is traditionally a manual, iterative process. This project automates parameter identification using two approaches:

1. **Physics-based inversion** — Newton's inversion method with a forward constitutive model minimizes an objective function, which is based on the misfit between predicted and observed stress-strain curves as well as constraining parameters within realistic physical bounds.
2. **Neural network surrogate** — a trained ensemble of MLPs predicts parameters directly from extracted curve features, validated against the Newton inversion results

The Mohr-Coulomb (MC) model is used as the primary constitutive model, applied to the publicly available Wichtmann Karlsruhe fine sand triaxial database. A key finding is that MC's limitations — stress-independent stiffness and inability to model plastic compression — are quantifiable through the inversion residuals.

The Modified Cam Clay (MCC) model was also used 
 
## USAGE 
**0. Run the Binder**
Click the Binder badge above to run the project directly in your browser — no installation needed. 
The steps below are only required if you prefer to run locally on your own machine.

**1. Clone the repository:**
```bash
git clone https://github.com/lambertnguyen-git/Constitutive-Soil-Models-Inversion-and-Neural-Networks
cd Constitutive-Soil-Models-Inversion-and-Neural-Networks
```
**2. Create and activate the environment:**
```bash 
conda env create -f environment.yml 
conda activate eosc-capstone
```
> WINDOWS USERS: you may have to adjust the environment file from python-mumps to -pydiso
**3. Run the inversion notebook.**
Open `ConstitutiveModel_Inversion.ipynb` and run all cells top to bottom.

Expected outputs:
- Per-test fit plots saved to `outputs/TMD#_plots.png`
- Convergence plots saved to `outputs/TMD#_iterations.png`
- Summary table printed to console
- Results saved to `outputs/MC_inversion_results.xlsx`

**4. Run the neural network notebook:**
Open `NN.ipynb` and run all cells top to bottom.

## NEXT STEPS
 

## ARTIFICIAL INTELLIGENCE USE
AI tools, including Claude (Anthropic) and the VS Code built-in AI assistant, were used to support code development and debugging throughout this project.
