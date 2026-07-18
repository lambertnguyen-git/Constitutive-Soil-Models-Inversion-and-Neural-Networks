# UBC EOSC '26 Capstone Project
## Neural Network-Assisted Determination and Inversion of Constitutive Model Parameters

Run this project in your browser:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lambertnguyen-git/Constitutive-Soil-Models-Inversion-and-Neural-Networks/HEAD?urlpath=%2Fdoc%2Ftree%2FInversion_MC.ipynb)

## REPORTING
This code is meant to accompany the report: Inversion and Neural Network Methods for Constitutive Model Parameter Identification, by L. Nguyen (July 2026).
References and citations are provided in that report.

## REPOSITORY COMPONENTS
1. Inversion_MC.ipynb: Mohr-Coulomb inversion workbook
2. Inversion_MCC.ipynb: Modified Cam-Clay inversion workbook
3. NeuralNetwork_MC.ipynb: Mohr-Coulomb neural network workbook
4. NeuralNetwork_MCC.ipynb: Modified Cam-Clay neural network workbook
5. project_utils.ipynb: Support workbook containing most of the main functions from the inversion and neural network workbooks. This helps reduce 'clutter' from the main workbooks. The other notebooks will call the required functions from project_utils in the initial import cell for each notebook.

## BACKGROUND AND APPROACH
Calibrating constitutive soil model parameters from laboratory triaxial test data is traditionally a manual, iterative process. This project automates parameter identification using two approaches:

1. **Physics-based Newton inversion** — Newton's inversion method with a forward constitutive model minimizes an objective function, which is based on the misfit between predicted and observed stress-strain curves as well as constraining parameters within realistic physical bounds.
2. **Neural network surrogate** — a trained ensemble of MLPs predicts parameters directly from extracted curve features, cross-validated against the Newton inversion results

The Mohr-Coulomb (MC) model is used as the primary constitutive model, applied to the publicly available Wichtmann Karlsruhe fine sand triaxial database. A key finding is that MC's limitations — stress-independent stiffness and inability to model plastic compression — are quantifiable through the inversion residuals.

The Modified Cam Clay (MCC) model is appropriate for contractive behaviour and was applied to the Wichtmann Karlsruhe fine sand as well. It had decent results for looser sand samples (TMD1-5) but broke down for denser samples.
 
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

**3. Run the inversion and neural network notebook.**
Open `Inversion_MC.ipynb`, `Inversion_MCC.ipynb`, `NeuralNetwork_MC.ipynb`, `NeuralNetwork_MCC.ipynb` and run all cells top to bottom.

Some of these cells are computationally intensive, such as the inversion, synthetic data generation, and neural network training. In some cases, we have saved the model or results at certain steps so that this lengthy processing can be avoided. Notes are provided which cells are required to run.

Expected outputs:
- Per-test fit plots saved to `outputs/TMD#_plots.png`
- Convergence plots saved to `outputs/TMD#_iterations.png`
- Summary table printed to console
- Results saved to `outputs/[model]_inversion_results.xlsx` and `outputs/[model]_NN_results.xlsx`


## ARTIFICIAL INTELLIGENCE USE
AI tools, including Claude (Anthropic) and the VS Code built-in AI assistant, were used to support code development and debugging throughout this project.
