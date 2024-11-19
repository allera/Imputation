#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:20:07 2021

@author: alblle
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

module_path='/Users/alblle/Dropbox/POSTDOC/EU_aims/github/imputation/imputation_module'
sys.path.append(os.path.abspath(module_path))
from  Imputation_EU_AIMS_behavioural_short  import *

def cli_parser():
    # Create a parser. It'll print the description on the screen.
    parser = argparse.ArgumentParser(description=__doc__)
    # Add a positional argument
    parser.add_argument('Infile', help=' Input csv file with missing values as 999. Variables in columns, Observations in rows.')
    # Add optional arguments
    parser.add_argument('-savedir', help='Output directory',default='/Users/alblle/Desktop/Imputation_Output')
    
    return parser

def main():     
    parser = cli_parser()
    # Get the inputs from the command line:
    args = vars(parser.parse_args())
    #import pdb;pdb.set_trace()
    imputation999(**args)

def imputation999(Infile=None, savedir='/Users/alblle/Desktop/Imputation_Output'): 

	
	
	
	#load missing data file
	df_orig,description_orig=load_csv_FINAL(Infile)
	df=df_orig.copy(deep=True)
	
	#make directory to save the imputed data
	if os.path.isdir(savedir) == 0:
		        os.mkdir(savedir)

	all_strategies=['Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression']
	all_params=[1, 2, 3]
	strategy= all_strategies[2]
	params=all_params[0]	    
	print('Performing ', strategy, 'inputation on file ', Infile )
	print('...')
	
	#extend to multi tab as in paper
	#tabs_path='/Users/alblle/Dropbox/POSTDOC/EU_aims/github/imputation/imputation_module'
	#order_of_inputation=[1,3,4,2,5,6,7,15,8,9,10,11,12,13,14]    
	#for tab in order_of_inputation:
	#Tab2inputate=tab            
	#mystr='Tab'+str(Tab2inputate)+'.csv'
	#path2missing=os.path.join(tabs_path,mystr)
        
		        

	X_missing=df.values
	imputed_data=my_inputer_FINAL(X_missing,strategy,params)  
	idx=np.argwhere(X_missing==999)
	for pos in idx:
		df.iat[pos[0],pos[1]]=imputed_data[pos[0],pos[1]]
		        
	#modify df_orig to save it, using the required variables from df  
	df_imputed=df_orig.copy(deep=True)
	variables_needed=df_imputed.columns 
	variables_df=df.columns
	for onevar in variables_needed:
		if onevar in variables_df:
			df_imputed[onevar]=df[onevar]
		        
	del df
	print('Input size =', df_imputed.shape)
	print('Output size =', df_orig.shape)
	file=os.path.basename(Infile)
	mystr=os.path.splitext(file)[0]+'_imputed.xlsx' #'Tab'+str(Tab2inputate)+'*_'+strategy+'_params'+str(params)+'.xlsx'
	path2save=os.path.join(savedir,mystr)
	print('Saving imputed data to ,', path2save )
	#df_imputed.to_csv(path_or_buf=path2save)

	df_imputed.to_excel(path2save)
	
	print('Enjoy')
	return df_imputed, df_orig

        
        
if __name__ == '__main__':
    main()


