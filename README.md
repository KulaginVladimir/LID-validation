# LID Validation

## Overview

This repository contains scripts to reproduce results on simulation of laser-induced desorption of deuterium from tungsten co-deposited films.

## Structure

The repository summaries the scripts to post-process raw experimental data and to perform LID simulations within different approximations.

[Experimental data](./experimental_data): This folder contains the scripts for the approximation of experimental laser pulses and post-processing of LID measurements.

[TDS](./TDS): This folder includes the experimental results of TDS measurements and the script for fitting of the measured spectrum to obtain the trap parameters for LID simulations.

[LID](./LID): This folder is dedicated to LID simulations: 

> [!WARNING]  
> Most of the scripts were ran on HPC with Slurm Workload Manager. The presented scripts were adapted for a sequential run on a local machine.
> For any queries, contact: VVKulagin@mephi.ru

  * [LID_1D](./LID/LID_1D): The input scripts for 1D simulations with varying value of the heat load.
  * [LID_1D](./LID/LID_1DD_2DT): The input scripts for 1D simulations of the D transport and 2D modelling of the heat transfer.
  * [LID_2D](./LID/LID_2D): The input scripts for 2D simulations.

[Comparison](./comparison): This folder contains the book for the final comparison between the results of modelling and experiments.

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