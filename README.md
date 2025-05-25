# DKDModel
Domain Knowledge-Driven Machine Learning Model for Performance Prediction and Structural Optimization of Solid Amine CO2 Adsorbents

This repo is the code base for the paper <paper>.
This study is a proof-of-concept; data is limited to peer-reviewed literature on **TEPA/silica solid amine adsorbents** for CO₂ capture.  


## Environment Setup
**Anaconda** is required to manage dependencies. The `far.yaml` file ensures reproducibility:  

1. Open **Anaconda Prompt** (Windows) or **terminal** (MacOS/Linux).  
2. Navigate to the repository's root directory.  
3. Run:  
   ```bash
   conda env create -f far.yaml
   conda activate far

## Workflow
Locate the `data.xlsx` file provided in the Supporting Information
Manually copy this file to: ./backends/

This repo has 2 main files:
- Run.py : Generates all 16 models
- Run_Ex_Data_Pre.py : Predicts the CO2 capacity of Ex-data
