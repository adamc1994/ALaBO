**ALaBO: Adaptive Latent Variable Bayesian Optimiser**

ALaBO is a MATLAB-based implementation of an Adaptive Latent Variable Bayesian Optimiser designed for efficient optimisation of chemical reactions involving both continuous and categorical variables. 
It combines latent variable Gaussian process (LVGP) modelling with an adaptive expected improvement (AEI) acquisition function to navigate complex, mixed-variable spaces effectively.

**Features**

Mixed Variable Optimisation: Handles both continuous and categorical variables seamlessly.

Adaptive Exploration-Exploitation Balance: Utilises AEI to dynamically adjust the trade-off between exploring new areas and exploiting known promising regions.

Latent Variable Modelling: Maps categorical variables into a continuous latent space, enabling the use of standard Gaussian process techniques.

Benchmarking: Demonstrated superior performance over other open-source Bayesian optimisation toolboxes in various catalytic reaction scenarios.

**Getting Started**

**Prerequisites**

MATLAB R2021a or later.

Statistics and Machine Learning Toolbox.

**Installation**

Clone the Repository:

git clone https://github.com/adamc1994/ALaBO.git

Add to MATLAB Path:

In MATLAB, navigate to the cloned directory and add it to your path:

addpath(genpath('path_to_cloned_ALaBO_directory'));

**Usage**

The main function, LVBayesianOptimiser.m, serves as the entry point for the optimiser. It requires defining the objective function, variable types, bounds, and other optimisation parameters.
For a detailed example, see the 'Test Function' folder, which contains the application of ALaBO to optimise a discretised form of the Branin function. This includes visualisation of the model and results. 

**Applications**

ALaBO has been successfully applied to the self-optimisation of catalytic reactions, including the Suzuki–Miyaura cross-coupling reaction, achieving high yields with a limited number of experiments (see reference below).
Its ability to handle mixed-variable spaces makes it suitable for various chemical engineering and process optimisation tasks.

**Reference**

If you use ALaBO in your research, please cite the following publication:

Aldulaijan, N., Marsden, J. A., Manson, J. A., & Clayton, A. D. (2024). Adaptive mixed variable Bayesian self-optimisation of catalytic reactions. Reaction Chemistry & Engineering, 9, 308–316. DOI: 10.1039/D3RE00476G

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or collaborations, please contact A.D.Clayton@leeds.ac.uk
