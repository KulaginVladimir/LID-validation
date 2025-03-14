# LID Validation

## Overview

This repository contains scripts to reproduce the results on simulation of laser-induced desorption of deuterium from tungsten co-deposited films, presented
at the OSSFE conference.

## Structure

The repository summaries the scripts to post-process raw experimental data and to perform LID simulations within different approximations.

[Experimental data](./experimental_data): This folder contains the scripts for the approximation of experimental laser pulses and post-processing of LID measurements.

[TDS](./TDS): This folder includes the experimental results of TDS measurements and the script for fitting of the measured spectrum to obtain the trap parameters for LID simulations.

[LID](./LID): This folder is dedicated to LID simulations. It includes three models within different approximations. 

> [!WARNING]  
> Most of the scripts were ran on HPC with Slurm Workload Manager. The presented scripts were adapted for a sequential run on a local machine.
> For any queries, contact: VVKulagin@mephi.ru

[Comparison](./comparison.ipynb): The book is for the final comparison between the results of modelling and experiments.

## How to use

For a local use, clone this repository to your local machine.

```
git clone https://github.com/KulaginVladimir/LID-validation.git
```

Create and activate the correct conda environment with the required dependencies:

```
conda env create -f environment.yml
conda activate LID_validation_env
```

This will set up a Conda environment named `LID_validation_env` with all the required dependencies for running the FESTIM scripts.

Navigate to the desired folder and run the Jupyter books using the activated Conda environment.

> [!NOTE]  
> LaTeX is required to reproduce the figures. To install required dependencies, run the following command in your terminal:
> ```
> sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
> ```
