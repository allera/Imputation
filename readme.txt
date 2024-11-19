Readme:

#How did I do (this is just for me to remember)
#create environment
#conda create --name imputation
#conda activate imputation
#conda install pandas
#conda install -c anaconda scikit-learn 
#conda install -c anaconda openpyxl
#conda env export > imputation_conda_env.yml



#How will YOU do !!!!!!!!!!!!
#cd to the directory where imputation_conda_env.yml is located
conda env create -f imputation_conda_env.yml
conda activate imputation




#run as
python imputation999.py -savedir '/Users/alblle/Desktop/Imputation_Output' 'Tab1.csv'


#or using the GUI as
python python imputation_GUI.py 


python imputation999_multitab.py -savedir '/Users/alblle/Desktop/Imputation_Output3' '/Users/alblle/Dropbox/POSTDOC/EU_aims/imputation_data_04_2022/phenos_for_imputation_T1_renamed'


  






