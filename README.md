# Imputation of Table data (.csv) missing values using extra trees regressors
https://pubmed.ncbi.nlm.nih.gov/35971088/

# Input : .csv fikle with missing values coded as 999
# Output : imputed data file i.e. same data but with no missing values


# create and activate the provided virtual environment
cd to the directory where imputation_conda_env.yml is located
conda env create -f imputation_conda_env.yml
conda activate imputation

# run as
python imputation999.py -savedir './Imputation_Output' 'Tab1_demo.csv'


# or using the GUI as
python imputation_GUI.py 

