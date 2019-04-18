# Machine Learning Pipeline with Delinquency Data

The goal of this assignment is to build a simple, modular, extensible, machine learning pipeline in Python. The pipeline should have functions that can do the following tasks:

1. Read/Load Data

2. Explore Data

3. Pre-Process and Clean Data

4. Generate Features/Predictors

5. Build Machine Learning Classifier

6. Evaluate Classifier

## Getting Started

I used conda as the environment manager.

### Prerequisites

1. Clone the repository.

        $ git clone https://github.com/capp-machine-learning/delinquency-ml-analysis-pipeline.git
        $ cd delinquency-ml-analysis-pipeline

1. A yml file of the environment is available in environment.yml.

        $ conda env create --file=environment.yml
        $ conda activate mlpipe
        
### Files

        delinquency-ml-analysis-pipeline
        ├── data
        │   ├── credit-data.csv
        │   ├── data-dictionary.xls
        │   └── tree.dot
        ├── config.py
        ├── environment.yml
        ├── ML_pipeline.ipynb
        ├── pipeline.py
        └── README.md

- __ML_pipeline.ipynb.ipynb__: This file is contains a writeup describing what I did and the results of running the code.
- __pipeline.py__: This python file has a collection of functions that I have written for this analysis. All functions are imported and used in __ML_pipeline.ipynb__.
- __config.py__: This python file contains the configuration for the code used for this analysis. These hardcoded values can be easily changed in the future for other uses.
- __data__: This folder contains the dataset and the data dictionary for this analysis.
