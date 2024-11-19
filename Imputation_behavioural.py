import pandas as pd
import os
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import sys
import pickle
#import scipy.stats

import warnings
warnings.filterwarnings("ignore")

def load_csv(path2csv):
    
    #load a csv file, all missing values are ssumed to be 999
    #In particular, it codes Control as 0 and ASD as 1
    
    missing_value=999
    data_file=path2csv

    #read the csv into pandas dataframe
    #print('loading the file ...',data_file)
    df = pd.read_csv(data_file, index_col=0)
    size=df.shape
    #print('Original data matrix size =',size)
    #r
    print("There are ", np.sum(df.values==999), "missing values", '(',np.sum(df.values==999)/np.prod(size),'%)')
    
    if 0:
        df.replace(to_replace=['Control','ASD'],value=[1,2],inplace=True)


    df2=df.copy()
    df2.replace(to_replace=[999],value=np.nan,inplace=True)
    description=df2.describe().transpose()
    #normalized_df2=(df2-description['mean'])/description['std']

    return df,description

def load_excel_FINAL(path2xlsx):
    
    #load a csv file, all missing values are ssumed to be 999
    #In particular, it codes Control as 0 and ASD as 1
    
    missing_value=999
    data_file=path2xlsx

    #read the csv into pandas dataframe
    #print('loading the file ...',data_file)
    df = pd.read_excel(data_file, index_col=0)
    size=df.shape
    #print('Original data matrix size =',size)
    #print("There are ", np.sum(df.values==999), "missing values", '(',np.sum(df.values==999)/np.prod(size),'%)')
    
    if 0:
        df.replace(to_replace=['Control','ASD'],value=[1,2],inplace=True)


    df2=df.copy()
    df2.replace(to_replace=[999],value=np.nan,inplace=True)
    description=df2.describe().transpose()
    #normalized_df2=(df2-description['mean'])/description['std']

    return df,description

def my_inputer(X_missing,strategy):
    #params can be 1,2 or 3, giving the possibility of using 3 differnt sets of parameters
    
    # To use this experimental feature, we need to explicitly ask for it:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.datasets import fetch_california_housing
    from sklearn.impute import SimpleImputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestRegressor

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    from sklearn.impute import MissingIndicator
    import numbers
    
    if (strategy == 'mean')|(strategy == 'median'): 
        # Estimate the score after imputation (mean and median strategies)
        model=0
    elif strategy == 'Extra_Trees_Regressor': #Extra trees regressor
        params=1
        if params==1:
            model=ExtraTreesRegressor(n_estimators=10, random_state=0)
        elif params==2:
            model=ExtraTreesRegressor(n_estimators=50, random_state=0)
        elif params==3:
            model=ExtraTreesRegressor(n_estimators=100, random_state=0)
       

    
    if isinstance(model, numbers.Number):
        imp = SimpleImputer(missing_values=999, strategy=strategy)
    else:
        imp=IterativeImputer(missing_values=999, random_state=0, estimator=model,max_iter=15,imputation_order='ascending' )
        #imputation_order : str, optional (default=?ascending?)

    imp.fit(X_missing)
    inputed_data=imp.transform(X_missing)
    
    return inputed_data

# def load_and_clean_xlsx_subset(path2xlsx):
    
#     #load a csv file and do some cleaning: code strings as numerical values and makes all missing values into 999
#     #In particular, it codes Control as 0 and ASD as 1
#     #returns a pandas dataframe with all missing values set to 999
    
#     missing_value=999
#     data_file=path2xlsx

#     #read the csv into pandas dataframe
#     print('loading the file ...',data_file)
#     df = pd.read_excel(data_file, index_col=0)
#     print('Original data matrix size =',df.shape)

#     print("There are ", np.sum(np.sum(df.isnull())), "non-coded missing values")

#     df.fillna(999, inplace=True) # , axis=0)

#     print("Substituting non-code missing values by 999, there are ", np.sum(np.sum(df.isnull())), "non-coded missing values left")

#     #Remove ugly variables and make all coded as missing (999 and 777) take the same value 999


#     #these variables having a fucked up coding....
#     #df.drop(['t1_iqtype','t1_drugclass_1','t1_drugclass_2','t1_drugclass_3','t1_handedness'],axis=1,inplace=True)
#     #print('Removing two variables, data matrix size =',df.shape)
    

#     #substitute non-numeric values by numbers...
#     #print(df['t1_diagnosis'].unique())
#     #df.replace(to_replace=['Control','ASD'],value=[0,1],inplace=True)
#     #df.replace(to_replace=['Adults','Children','Adolescents','Adol. & adults (IQ <75)'],value=[3,1,2,0],inplace=True)
#     #df.replace(to_replace=[777],value=[999],inplace=True)
    
    
#     if 0:
#         df.replace(to_replace=['Control','ASD'],value=[1,2],inplace=True)


#     df2=df.copy()
#     df2.replace(to_replace=[999],value=np.nan,inplace=True)
#     description=df2.describe().transpose()
#     #normalized_df2=(df2-description['mean'])/description['std']

    
#     return df,description




# def load_and_clean_xlsx(path2xlsx):
    
#     #load a csv file and do some cleaning: code strings as numerical values and makes all missing values into 999
#     #In particular, it codes Control as 0 and ASD as 1
#     #returns a pandas dataframe with all missing values set to 999
    
#     missing_value=999
#     data_file=path2xlsx

#     #read the csv into pandas dataframe
#     print('loading the file ...',data_file)
#     df = pd.read_excel(data_file, index_col=0)
#     print('Original data matrix size =',df.shape)

#     print("There are ", np.sum(np.sum(df.isnull())), "non-coded missing values")

#     df.fillna(999, inplace=True)

#     print("Substituting non-code missing values by 999, there are ", np.sum(np.sum(df.isnull())), "non-coded missing values left")

#     #Remove ugly variables and make all coded as missing (999 and 777) take the same value 999


#     #these variables having a fucked up coding....
#     df.drop(['t1_iqtype','t1_drugclass_1','t1_drugclass_2','t1_drugclass_3','t1_handedness'],axis=1,inplace=True)
#     #print('Removing two variables, data matrix size =',df.shape)
    

#     #substitute non-numeric values by numbers...
#     #print(df['t1_diagnosis'].unique())
#     #df.replace(to_replace=['Control','ASD'],value=[0,1],inplace=True)
#     #df.replace(to_replace=['Adults','Children','Adolescents','Adol. & adults (IQ <75)'],value=[3,1,2,0],inplace=True)
#     df.replace(to_replace=[777],value=[999],inplace=True)

#     #df.replace(to_replace=[777777],value=[999],inplace=True)

#     return df



# def split_df(df):
#     #split pandas data frame in groups:
#     #returns a data frame with only ASD data, a dataframe with only controls data

#     missing_value=999
#     df.replace(to_replace=['Control','ASD'],value=[1,2],inplace=True)
#     N_controls=np.sum(df['t1_diagnosis']==1)
#     N_asd=np.sum(df['t1_diagnosis']==2)

#     print('Total number of subjects = ', N_asd + N_controls)
#     print('Number of controls = ', N_controls)
#     print('Number of diagnosed ASD = ', N_asd)


#     #make 2 data frames, one for controls, one for ASD

#     asd_df=df[df['t1_diagnosis']==2]
#     ct_df=df[df['t1_diagnosis']==1]

#     #Identify which variables were gathered "for all subjects" and which ones only gathered for ASD subjects  
#     all_subjects_measures=[]
#     just_ASD_measures=[];
#     for col in df.columns:
#         #print(col)
#         if np.sum(ct_df[col]==missing_value) == N_controls:
#             just_ASD_measures.append(col)
#         else:
#             all_subjects_measures.append(col)    


#     print('Split data in two subsets by splitting the variables in two sets \
#     one containing the behavioral measures only gathered for ASD subjects')
    
#     asd_measures_df=asd_df[just_ASD_measures]
#     all_measures_df=df[all_subjects_measures] #[df['t1_diagnosis']==0]

#     print('ASD measures data matrix', asd_measures_df.shape)
#     print('all subjects measures data matrix', all_measures_df.shape)

#     asd_subjects_df=asd_df
#     control_subjects_df=ct_df
    
#     return asd_measures_df, all_measures_df, asd_subjects_df, control_subjects_df


# def stats_on_missing_data(data_df):

#     missing_value=999
#     data_np=data_df.values
#     n,p =data_np.shape # n is number of observations (subjects), p is number of variables
#     data_missing_values=(data_np==missing_value)
#     #np.sum(data_missing_values,0)

#     available_observations = dict() #dictionary
#     available_number_observations = dict() #dictionary
#     available_percentage_observations = dict() #dictionary

#     for variable_idx in range(p):
#         available_observations.update({data_df.columns[variable_idx]: data_np[:,variable_idx] != missing_value}) 
#         available_number_observations.update({data_df.columns[variable_idx]: np.sum(data_np[:,variable_idx] != missing_value)}) 
#         available_percentage_observations.update({data_df.columns[variable_idx]: 100 * (np.sum(data_np[:,variable_idx] != missing_value)/n)})

#     return available_observations,available_number_observations,available_percentage_observations



# def select_measures(data_df,available_percentage_observations,percentage_threshold):
#     #data_df is a data frame, the same used as input to stats_on_missing_data
#     #available_percentage_observations is output from stats_on_missing_data
#     #percentage_threshold is used to remove measures not available for less than such percentage of subjects

#     some_dict=available_percentage_observations
#     values = []
#     for key in some_dict:
#         values.append(some_dict[key])

#     #keep measures for which there is enough data

#     #percentage_threshold=70
#     subjects_selected_measures=[]
#     idx=-1
#     for col in data_df.columns:
#         idx=idx+1
#         if np.asarray(values)[idx] >percentage_threshold:
#             subjects_selected_measures.append(col)

#     df_selected_measures=data_df[subjects_selected_measures]
#     print('Selected measures data matrix', df_selected_measures.shape)

#     return df_selected_measures, values

# def build_sub_dataset_full_observations(df_selected_measures):
#     #df_selected_measures is usually output of select_measures
#     #selects the subset of subjects for which all variables are observed
    
#     good_subjects = np.ndarray.flatten(np.argwhere(np.max(df_selected_measures.values,1)!=999))
#     df_selected_measures_subjects=df_selected_measures.iloc[good_subjects]

#     print('keeping only subjects with all selected observations')
#     print('Selected measures data matrix', df_selected_measures_subjects.shape)

#     return df_selected_measures_subjects

# def imputations_comparison(my_df):
#     # To use this experimental feature, we need to explicitly ask for it:
#     from sklearn.experimental import enable_iterative_imputer  # noqa
#     from sklearn.datasets import fetch_california_housing
#     from sklearn.impute import SimpleImputer
#     from sklearn.impute import IterativeImputer
#     from sklearn.linear_model import BayesianRidge
#     from sklearn.tree import DecisionTreeRegressor
#     from sklearn.ensemble import ExtraTreesRegressor
#     from sklearn.neighbors import KNeighborsRegressor
#     from sklearn.pipeline import make_pipeline
#     from sklearn.model_selection import cross_val_score
#     from sklearn.metrics import mean_squared_error
#     from sklearn.impute import MissingIndicator
#     import numbers


#     my_df=(my_df-my_df.mean())/my_df.std()
#     X_full=my_df.values 
#     print('data matrix to input shape',X_full.shape)
#     n_samples, n_features = X_full.shape

#     N_SPLITS = 100
#     rng = np.random.RandomState(0)

#     #scores = pd.DataFrame()
#     e=dict.fromkeys(['mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])

#     for fold in range(N_SPLITS):
#         X_missing = X_full.copy()
#         #missing_samples = np.arange(n_samples)
#         #missing_features = rng.choice(n_features, n_samples, replace=True)
#         #X_missing[missing_samples, missing_features] = 999 #np.nan

#         randomized_subjects=np.random.permutation(n_samples);
#         missing_samples=randomized_subjects[range(n_features)]
#         missing_features=np.random.permutation(n_features);
#         X_missing[missing_samples, missing_features] = 999 #np.nan
        
#         for strategy in ('mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'):

        
#             if (strategy == 'mean')|(strategy == 'median'): 
#                 # Estimate the score after imputation (mean and median strategies)
#                 model=0
#             elif strategy == 'Bayesian_Ridge_Regression': #Bayesian Ridge Regression
#                 model=BayesianRidge()
#             elif strategy == 'Decision_Tree_Regression': #Decision tree regression
#                 model=DecisionTreeRegressor(max_features='sqrt', random_state=0)
#             elif strategy == 'Extra_Trees_Regressor': #Extra trees regressor
#                 model=ExtraTreesRegressor(n_estimators=10, random_state=0)
#             elif strategy == 'Kneighbours_Regression': #Kneighbours regression
#                 model=KNeighborsRegressor(n_neighbors=15)
            
#             if isinstance(model, numbers.Number):
#                 imp = SimpleImputer(missing_values=999, strategy=strategy)
#             else:
#                 imp=IterativeImputer(missing_values=999, random_state=0, estimator=model)

#             imp.fit(X_missing)
#             inputed_data=imp.transform(X_missing)
#             #e.append(mean_squared_error(X_full[missing_samples, missing_features], inputed_data[missing_samples, missing_features]))
#             error=mean_squared_error(X_full[missing_samples, missing_features], inputed_data[missing_samples, missing_features])
            
#             if fold==0:
#                 e[strategy]=error
#             else:
#                 e[strategy]=np.append(e[strategy],error)

#     scores= pd.DataFrame.from_dict(e)

#     return scores



# def make_scores_plot(scores,my_title,savefig):
    
#     fig, ax = plt.subplots(figsize=(13, 6))
#     means = scores.mean()
#     errors = scores.std()
#     means.plot.barh(xerr=errors, ax=ax)
#     ax.set_title(my_title)
#     ax.set_xlabel('MSE (smaller is better)')
#     ax.set_yticks(np.arange(means.shape[0]))
#     ax.set_yticklabels([label for label in means.index.get_values()])
#     #ax.set_xlim(0,2)

#     plt.tight_layout(pad=1)
    
#     if savefig==0:
#         plt.show()
#     else:        
#         plt.savefig(my_title)

#     return

# def plot_percentages_availabel(values,values_asd,savefig):
#     fig,axs=plt.subplots(1,2,figsize=(20,10),sharey=True)
#     axs[0].plot(values,label='orig. order')
#     axs[0].plot(sorted(values,reverse=True),label='sorted order')
#     axs[0].set_title('all subjects')
#     axs[0].set_xlabel('Measures')
#     axs[0].set_ylabel('Percentage available data')
#     axs[0].legend(loc="upper right")



#     axs[1].plot(values_asd,label='orig. order')
#     axs[1].plot(sorted(values_asd,reverse=True),label='sorted order')
#     axs[1].set_title('ASD subjects')
#     axs[1].set_xlabel('Measures')
#     axs[1].set_ylabel('Percentage available data')
#     axs[1].legend(loc="upper right")

#     if savefig==0:
#         plt.show()
#     else:        
#         plt.savefig('available_percentage.png')

#     return
#     #fig.savefig('available_percentage.png')
    


    


# def plot_correlation_structures(asd_df,ct_df,savefig):
#     #Plot the correlation structures
#     asd_corr = asd_df.corr()
#     ct_corr = ct_df.corr()
#     _, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10)) 

#     # plot a correlation matrix
#     tmp_ax = sns.heatmap(asd_corr, ax=ax1,cmap='coolwarm',vmin=-1,vmax=1)
#     for _, spine in tmp_ax.spines.items():
#         spine.set_visible(True)

#     tmp_ax.set_title('ASD subjects')
            
#     tmp_ax = sns.heatmap(ct_corr, ax=ax2,cmap='coolwarm',vmin=-1,vmax=1)  
#     for _, spine in tmp_ax.spines.items():
#         spine.set_visible(True)
        
#     tmp_ax.set_title('Control subjects')

#     if savefig==0:
#         plt.show()
#     else:        
#         plt.savefig('correlations.png')

#     return


# def dict2list(some_dict):
# #[values keys]=dict2list(all_available_percentage_observations)
#     values = []
#     keys = []
#     idx=-1
#     for key in some_dict:
#         idx=idx+1
#         values.append(some_dict[key])
#         keys.append(key)
        
# #        if idx==0:
# #            npvalues=np.zeros([some_dict[key].shape[0],len(some_dict.keys())])
# #        
# #        npvalues[:,idx]=some_dict[key]
        
#     return values,keys #,npvalues


# def imputations_comparison2(input_df,imputation_strategies,standardize):
     
#     variable_names=input_df.columns
#     N_variables=variable_names.shape[0]
#     available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     [values, keys]=dict2list(available_percentage_observations)
    
#     #sort the variabls accoriding to decreasing or increasing number of observations
#     #argsort_idx=(-np.asarray(values)).argsort()
#     #reorder input as decreasing number of observations
#     #input_df=input_df[variable_names[argsort_idx]]
#     #variable_names=input_df.columns
#     #N_variables=variable_names.shape[0]
#     #available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     #[values, keys]=dict2list(available_percentage_observations)

#     #my_df=(my_df-my_df.mean())/my_df.std()
#     #standardize=0
#     if standardize==1:
#         X_init=np.copy(input_df.values)
#         X_non_std=np.copy(input_df.values)

#         #change 999 by nan to can comoute means and stds easier...
#         input_df2=input_df.copy()
#         input_df2.replace(to_replace=[999],value=np.nan,inplace=True)
#         summary_stats=input_df2.describe().transpose() 
#         standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#         X_std=np.copy(standard_df.values) 
#         #put back the 999 coding
#         X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]

#         X_init=X_std
        

#     else:
#         X_init=np.copy(input_df.values)

        
#     X_init_back_up=X_init.copy()
#     X_filled=X_init.copy()
#     X_filled_back_up=X_filled.copy()

    
#     print('data matrix to input shape',X_init.shape)
#     n_samples, n_features = X_init.shape
#     #N_SPLITS = 100
#     #rng = np.random.RandomState(0)

#     #scores = pd.DataFrame()
#     #e=dict.fromkeys(['mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])


#     #imputation_strategies=['mean', 'median']#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression']
    
    
#     all_errors=dict.fromkeys(imputation_strategies)
#     all_X_filled=dict.fromkeys(imputation_strategies)
#     all_X_reimputing_good_values=dict.fromkeys(imputation_strategies)
    
#     #if standardize==1:
#     all_errors_original_std=dict.fromkeys(imputation_strategies)

    
#     for strategy in imputation_strategies: # ('mean', 'median'):#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'):
        
#         all_errors[strategy]=dict.fromkeys(pd.Index.tolist(variable_names))
#         all_errors_original_std[strategy]=dict.fromkeys(pd.Index.tolist(variable_names))

#         X_init=X_init_back_up.copy()
#         X_filled=X_filled_back_up.copy()
#         X_reimputing_good_values=999 * np.ones(X_init.shape)
        
#         for col in range(N_variables): 
            
#             variable=variable_names[col]
            
#             print('XXXXXXXXXXXXXXXXXXXXXXXX  playing  ', variable, 'XXXXXXXXXXXXXXXXXXXXXX')
#             #all_errors[variable]=dict.fromkeys(imputation_strategies)#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])

#             good_subjects = np.ndarray.flatten(np.argwhere(input_df[variable].values !=999))
#             missing_subjects=np.ndarray.flatten(np.argwhere(input_df[variable].values == 999))
#             #print('variable', variable, 'has', good_subjects.shape[0], 'observations (subjects)')
            
#             #substitute observed values by 999 for testing
#             for sub in range(good_subjects.shape[0]):
                
#                 if col ==0:
#                     X_missing = X_init.copy()
#                 else:
#                     X_missing = X_filled.copy()
                   
                    
#                 X_missing[good_subjects[sub], col] = 999 
                
#                 inputed_data = my_inputer(X_missing,strategy)
                            
#                 error=np.sqrt(np.power(X_filled[good_subjects[sub], col] - inputed_data[good_subjects[sub], col],2))
                
#                 if standardize ==1:
#                     error_original=np.sqrt(np.power(X_non_std[good_subjects[sub], col] - np.add(np.multiply(inputed_data[good_subjects[sub], col],summary_stats['std'][col]),summary_stats['mean'][col]),2))
#                     #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        

                
#                 tmp_value=inputed_data[good_subjects[sub], col]
                
#                 #print(tmp_value)
                
#                 #print(X_reimputing_good_values[good_subjects[sub], col])
                
#                 X_reimputing_good_values[good_subjects[sub], col] = tmp_value
                
#                 #print(X_reimputing_good_values[good_subjects[sub], col])

                
#                 if sub==0:
#                     all_errors[strategy][variable]=error
#                     if standardize==1:
#                         all_errors_original_std[strategy][variable]=error_original
#                 else:
#                     all_errors[strategy][variable]=np.append(all_errors[strategy][variable],error)
#                     if standardize==1:
#                         all_errors_original_std[strategy][variable]=np.append(all_errors_original_std[strategy][variable],error_original)
                    
#             #impute the variable without adding dummy missing values
#             if col ==0:
#                 X_missing = X_init.copy()
#             else:
#                 X_missing = X_filled.copy()
                   
#             inputed_data = my_inputer(X_missing,strategy)

#             X_filled[:,col] = inputed_data[:,col]
#             1

#         if standardize==0:
#             all_X_filled[strategy]=X_filled
#             all_X_reimputing_good_values[strategy]=X_reimputing_good_values
#         else:
#             all_X_filled[strategy]= np.add(np.multiply(X_filled,np.asarray(summary_stats['std'])) , np.asarray(summary_stats['mean']))
            
#             all_X_reimputing_good_values[strategy]=np.add(np.multiply(X_reimputing_good_values,np.asarray(summary_stats['std'])), np.asarray(summary_stats['mean']))
#             all_X_reimputing_good_values[strategy][np.where(X_init==999)]=999
#             #X_init[np.where(X_init==999)]
#             1
#             #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#             #X_std=standard_df.values 
#             ##put back the 999 coding
#             #X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]
#             #X_init=X_std
                
        
            
            
            

#     #scores= pd.DataFrame.from_dict(e)
#     initial_data = input_df.values
#     return all_errors, variable_names, all_X_filled, all_X_reimputing_good_values,initial_data, all_errors_original_std


# def my_inputer(X_missing,strategy):
#      # To use this experimental feature, we need to explicitly ask for it:
#     from sklearn.experimental import enable_iterative_imputer  # noqa
#     from sklearn.datasets import fetch_california_housing
#     from sklearn.impute import SimpleImputer
#     from sklearn.impute import IterativeImputer
#     from sklearn.linear_model import BayesianRidge
#     from sklearn.tree import DecisionTreeRegressor
#     from sklearn.ensemble import ExtraTreesRegressor
#     from sklearn.ensemble import RandomForestRegressor

#     from sklearn.neighbors import KNeighborsRegressor
#     from sklearn.pipeline import make_pipeline
#     from sklearn.model_selection import cross_val_score
#     from sklearn.metrics import mean_squared_error
#     from sklearn.impute import MissingIndicator
#     import numbers
    
#     if (strategy == 'mean')|(strategy == 'median'): 
#         # Estimate the score after imputation (mean and median strategies)
#         model=0
#     elif strategy == 'Bayesian_Ridge_Regression': #Bayesian Ridge Regression
#         model=BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=10,
#        normalize=False, tol=0.001, verbose=False)
#     elif strategy == 'Decision_Tree_Regression': #Decision tree regression
#         model=DecisionTreeRegressor(max_features='sqrt', random_state=0)
#     elif strategy == 'Extra_Trees_Regressor': #Extra trees regressor
#         model=ExtraTreesRegressor(n_estimators=10, random_state=0)
#     elif strategy == 'Kneighbours_Regression': #Kneighbours regression
#         model=KNeighborsRegressor(n_neighbors=15)
#     elif strategy ==  'RandomForestRegressor':
#         model=RandomForestRegressor(n_estimators = 100, random_state = 0)
    
#     if isinstance(model, numbers.Number):
#         imp = SimpleImputer(missing_values=999, strategy=strategy)
#     else:
#         imp=IterativeImputer(missing_values=999, random_state=0, estimator=model,max_iter=15,imputation_order='ascending' )
#         #imputation_order : str, optional (default=?ascending?)

#     imp.fit(X_missing)
#     inputed_data=imp.transform(X_missing)
    
#     return inputed_data

# def imputations_convergence_test(input_df,imputation_strategies):
     
#     variable_names=input_df.columns
#     N_variables=variable_names.shape[0]
#     available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     [values, keys]=dict2list(available_percentage_observations)
    
#     #sort the variabls accoriding to decreasing or increasing number of observations
#     argsort_idx=(-np.asarray(values)).argsort()
#     #argsort_idx=(np.asarray(values)).argsort()
    
    
#     #reorder input as decreasing number of observations
#     input_df=input_df[variable_names[argsort_idx]]
#     variable_names=input_df.columns
#     N_variables=variable_names.shape[0]
#     available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     [values, keys]=dict2list(available_percentage_observations)

#     #my_df=(my_df-my_df.mean())/my_df.std()
#     X_init=input_df.values 
#     X_filled=X_init.copy()
    
#     X_init_back_up=X_init.copy()
#     X_filled_back_up=X_filled.copy()

    
#     print('data matrix to input shape',X_init.shape)
#     n_samples, n_features = X_init.shape
#     #N_SPLITS = 100
#     #rng = np.random.RandomState(0)

#     #scores = pd.DataFrame()
#     #e=dict.fromkeys(['mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])


#     #imputation_strategies=['mean', 'median']#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression']
    
    
#     all_errors=dict.fromkeys(imputation_strategies)

    
#     for strategy in imputation_strategies: # ('mean', 'median'):#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'):
        
#         all_errors[strategy]=dict.fromkeys(pd.Index.tolist(variable_names))

#         X_init=X_init_back_up.copy()
#         X_filled=X_filled_back_up.copy()
        
#         for col in range(N_variables): 
            
#             variable=variable_names[col]
#             #all_errors[variable]=dict.fromkeys(imputation_strategies)#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])

#             good_subjects = np.ndarray.flatten(np.argwhere(input_df[variable].values !=999))
#             missing_subjects=np.ndarray.flatten(np.argwhere(input_df[variable].values == 999))
#             #print('variable', variable, 'has', good_subjects.shape[0], 'observations (subjects)')
            
                    
#             #impute the variable without adding dummy missing values
#             if col ==0:
#                 X_missing = X_init.copy()
#             else:
#                 X_missing = X_filled.copy()
                   
#             inputed_data = my_inputer(X_missing,strategy)

#             X_filled[:,col] = inputed_data[:,col]
#             1
            
            
            

#     #scores= pd.DataFrame.from_dict(e)

#     return all_errors, variable_names



# def Evaluate_one_imputer_function_FINAL(path2csv,dict_filled_paths,imputation_strategy,standardize,output_dir):
    
#     len(dict_filled_paths)

#     if os.path.isdir(output_dir) == 0:
#         os.mkdir(output_dir)
#     if os.path.isdir(os.path.join(output_dir,imputation_strategy[0])) == 0:
#         os.mkdir(os.path.join(output_dir,imputation_strategy[0]))
#     ########################################################################################
#     #Load data and split into variables acquired for all subjects and variables acquired only for ASD subjects.
#     #path2xlsx='/home/mrstats/petmul/Alberto/EU_aims/beh_data/julian_subset_measures/LEAP_t1_clinical%20variables_03-09-19-withlabels.xlsx'
#     df,description=load_csv_FINAL(path2csv) #loads csv 
#     #asd_measures_df, all_measures_df, asd_subjects_df, control_subjects_df = split_df(df)
    
#     ########################################################################################
#     #           IMPUTATION EVALUATION
    
#     input_df=pd.DataFrame.drop(df,labels='t1_diagnosis',axis=1)
#     errors, variable_names, X_filled, X_reimputing_good_values, input_data_reordered,errors_original_std=imputations_comparison2(input_df,imputation_strategy,standardize)


#     if os.path.isdir(os.path.join(output_dir,imputation_strategy[0],'ALL_measures')) == 0: 
#         os.mkdir(os.path.join(output_dir,imputation_strategy[0],'ALL_measures'))
        
#     if standardize ==0:
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_raw.p"), 'wb') as handle:
#             pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_filled_raw.p"), 'wb') as handle:
#             pickle.dump(X_filled, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_reimputing_good_values_raw.p"), 'wb') as handle:
#             pickle.dump(X_reimputing_good_values, handle, protocol=pickle.HIGHEST_PROTOCOL)     
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_input_data_reordered_raw.p"), 'wb') as handle:
#             pickle.dump(input_data_reordered, handle, protocol=pickle.HIGHEST_PROTOCOL)    
	      
		    
#     else:
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_std.p"), 'wb') as handle:
#             pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_filled_std.p"), 'wb') as handle:
#             pickle.dump(X_filled, handle, protocol=pickle.HIGHEST_PROTOCOL) 
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_reimputing_good_values_std.p"), 'wb') as handle:
#             pickle.dump(X_reimputing_good_values, handle, protocol=pickle.HIGHEST_PROTOCOL)   
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_input_data_reordered_std.p"), 'wb') as handle:
#             pickle.dump(input_data_reordered, handle, protocol=pickle.HIGHEST_PROTOCOL)       
#         with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_std_reconstructed.p"), 'wb') as handle:
#             pickle.dump(errors_original_std, handle, protocol=pickle.HIGHEST_PROTOCOL)     
    	 

    
            
	    
	    
#     return df



# def Evaluate_one_imputer_function_one_variable_FINAL(df,imputation_strategy,standardize,output_dir,variable2test,params):

    
#     #len(dict_filled_paths)

#     #if os.path.isdir(output_dir) == 0:
#     #    os.mkdir(output_dir)
#     #if os.path.isdir(os.path.join(output_dir,imputation_strategy[0])) == 0:
#     #    os.mkdir(os.path.join(output_dir,imputation_strategy[0]))
#     ########################################################################################
#     #Load data and split into variables acquired for all subjects and variables acquired only for ASD subjects.
#     #path2xlsx='/home/mrstats/petmul/Alberto/EU_aims/beh_data/julian_subset_measures/LEAP_t1_clinical%20variables_03-09-19-withlabels.xlsx'
#     #df,description=load_csv_FINAL(path2csv) #loads csv 
#     #Remove diagnosis from imputation to not bias regressions towards the label to predict, diagnosis
#     #if variable2test != 't1_diagnosis':
#     #    df=pd.DataFrame.drop(df,labels='t1_diagnosis',axis=1) 
        

#     ########################################################################################
#     #           IMPUTATION EVALUATION
    
          
            
    
        
#     errors, variable_names, X_filled, X_reimputing_good_values, input_data_reordered,errors_original_std=imputations_comparison_one_variable_FINAL(df,imputation_strategy,standardize,variable2test,params)    
#     #errors, variable_names, X_filled, X_reimputing_good_values, input_data_reordered,errors_original_std=imputations_comparison2(input_df,imputation_strategy,standardize)


    
#     if standardize ==0:
#         with open(os.path.join(output_dir,"all_errors_raw.p"), 'wb') as handle:
#             pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#         with open(os.path.join(output_dir,"all_X_filled_raw.p"), 'wb') as handle:
#             pickle.dump(X_filled, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open(os.path.join(output_dir,"all_X_reimputing_good_values_raw.p"), 'wb') as handle:
#             pickle.dump(X_reimputing_good_values, handle, protocol=pickle.HIGHEST_PROTOCOL)     
#         with open(os.path.join(output_dir,"all_input_data_reordered_raw.p"), 'wb') as handle:
#             pickle.dump(input_data_reordered, handle, protocol=pickle.HIGHEST_PROTOCOL)    
	       
#      # if os.path.isdir(os.path.join(output_dir,imputation_strategy[0],'ALL_measures')) == 0: 
#      #    os.mkdir(os.path.join(output_dir,imputation_strategy[0],'ALL_measures'))
           
#     # if standardize ==0:
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_raw.p"), 'wb') as handle:
#     #         pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)    
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_filled_raw.p"), 'wb') as handle:
#     #         pickle.dump(X_filled, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_reimputing_good_values_raw.p"), 'wb') as handle:
#     #         pickle.dump(X_reimputing_good_values, handle, protocol=pickle.HIGHEST_PROTOCOL)     
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_input_data_reordered_raw.p"), 'wb') as handle:
#     #         pickle.dump(input_data_reordered, handle, protocol=pickle.HIGHEST_PROTOCOL)    
# 	   #    
# 		  #   
#     # else:
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_std.p"), 'wb') as handle:
#     #         pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_filled_std.p"), 'wb') as handle:
#     #         pickle.dump(X_filled, handle, protocol=pickle.HIGHEST_PROTOCOL) 
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_X_reimputing_good_values_std.p"), 'wb') as handle:
#     #         pickle.dump(X_reimputing_good_values, handle, protocol=pickle.HIGHEST_PROTOCOL)   
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_input_data_reordered_std.p"), 'wb') as handle:
#     #         pickle.dump(input_data_reordered, handle, protocol=pickle.HIGHEST_PROTOCOL)       
#     #     with open(os.path.join(output_dir,imputation_strategy[0],'ALL_measures',"all_errors_std_reconstructed.p"), 'wb') as handle:
#     #         pickle.dump(errors_original_std, handle, protocol=pickle.HIGHEST_PROTOCOL)     
    	       
 

#     return 1

# def gather_results_fun(main_path,standardize):
#     measures_sets=['ALL_measures','ASD_measures']
#     #standarsize=0
    
#     models_2_compare=['mean','median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor', 'Kneighbours_Regression']#,'RandomForestRegressor']
    
#     if standardize==1:
#         preproc=['standardize']
#     else:
#         preproc=['raw']
    
#     saved_stats=['errors','input_data_reordered','filled_data','reimputed_good_values','errors_std_reconstructed']
    
    
#     all_info=dict.fromkeys(measures_sets)
#     # all_info[group]=dict.fromkeys(preproc)  
#     # all_info[group][preproc_opt]= dict.fromkeys(models_2_compare)
#     # errors=pickle.load(open(the_path1,'rb'))
#     # all_info[group][preproc_opt][model]['errors']=errors[model]

#     # #[values1, variables1]= dict2list(errors[model])

    
#     # input_data=pickle.load(open(the_path2,'rb'))
#     # all_info[group][preproc_opt][model]['input_data_reordered']=input_data
    
#     # filled_data=pickle.load(open(the_path3,'rb'))
#     # all_info[group][preproc_opt][model]['filled_data']=filled_data[model]
    
                
                
#                 # reimputed_good_values=pickle.load(open(the_path4,'rb'))
#                 # all_info[group][preproc_opt][model]['reimputed_good_values']=reimputed_good_values[model]
    
    
#     for group in measures_sets:
#         all_info[group]=dict.fromkeys(preproc)  
#         for preproc_opt in preproc:
#             all_info[group][preproc_opt]= dict.fromkeys(models_2_compare)
#             for model in models_2_compare:
                
#                 all_info[group][preproc_opt][model]=dict.fromkeys(saved_stats)
    
#                 tmp_path=os.path.join(main_path,model)
#                 main_path2=os.path.join(tmp_path,group)
                
#                 if group=='ALL_measures':
#                     prefix='all'
#                 else:
#                     prefix='asd'
                    
#                 if preproc_opt=='standardize':
#                     suffix='std'
#                     suffix2='std_reconstructed'
#                 else:
#                     suffix='raw'
                    
#                 the_path1=os.path.join(main_path2,prefix + "_errors_" + suffix + ".p")
#                 the_path2=os.path.join(main_path2,prefix + "_input_data_reordered_" + suffix + ".p")
#                 the_path3=os.path.join(main_path2,prefix + "_X_filled_" + suffix + ".p")
#                 the_path4=os.path.join(main_path2,prefix + "_X_reimputing_good_values_" + suffix + ".p")
#                 if preproc_opt=='standardize':#standardize==1:
#                     the_path5=os.path.join(main_path2,prefix + "_errors_" + suffix2 + ".p")
                
#                 errors=pickle.load(open(the_path1,'rb'))
#                 all_info[group][preproc_opt][model]['errors']=errors[model]
    
#                 #[values1, variables1]= dict2list(errors[model])
    
                
#                 input_data=pickle.load(open(the_path2,'rb'))
#                 all_info[group][preproc_opt][model]['input_data_reordered']=input_data
                
#                 filled_data=pickle.load(open(the_path3,'rb'))
#                 all_info[group][preproc_opt][model]['filled_data']=filled_data[model]
    
                
                
#                 reimputed_good_values=pickle.load(open(the_path4,'rb'))
#                 all_info[group][preproc_opt][model]['reimputed_good_values']=reimputed_good_values[model]
    
#                 1#['errors','input_data_reordered','filled_data','reimputed_good_values']
#                 if preproc_opt=='standardize':#standardize == 1:
#                     errors_std_reconstructed=pickle.load(open(the_path5,'rb'))
#                     all_info[group][preproc_opt][model]['errors_std_reconstructed']=errors_std_reconstructed[model]
#                 #[values2, variables2]= dict2list(all_errors2)
#                 #[values3, variables3]= dict2list(all_errors3)
#                 #[values4, variables4]= dict2list(all_errors4)
#     return all_info  


# def find_best_fun(all_info,preproc_opt):
#     measures_sets=['ALL_measures','ASD_measures']
#     models_2_compare=['mean','median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor', 'Kneighbours_Regression','RandomForestRegressor']

#     best_models=dict.fromkeys(measures_sets)
#     error_best_models=dict.fromkeys(measures_sets)
#     for group in measures_sets: #asd or all
#             variables=list(all_info[group][preproc_opt]['mean']['errors'])
#             best_models[group]=dict.fromkeys(variables)
#             error_best_models[group]=dict.fromkeys(variables)
#             for var in variables:
#                 mod_idx=-1
#                 for model in models_2_compare: 
#                     mod_idx=mod_idx+1
#                     if preproc_opt == 'raw':
#                         e=all_info[group][preproc_opt][model]['errors'][var]
#                     elif preproc_opt == 'standardize':
#                         e=all_info[group][preproc_opt][model]['errors_std_reconstructed'][var]
#                     if mod_idx==0:
#                         E=np.zeros([len(models_2_compare),e.shape[0]])
#                     E[mod_idx,:]=e
#                         #print('variable = ',var, ',model = ', model, ',size = ',e[model].shape)
#                 m=np.mean(E,1)
#                 s=np.std(E,1)
#                 best_model_idx=np.argmin(m)
#                 error_best_models[group][var]=m[best_model_idx]
                
#                 best_models[group][var]=models_2_compare[best_model_idx]
#                 print('variable = ',var, ', best model is ',best_models[group][var])
            
#                 #for i in range(5):
#                 #    for j in range(5):
#                 #        tmp=scipy.stats.ttest_rel(E[i,:], E[j,:])
#                 #        1
#     return best_models, error_best_models



# def imputations_comparison_one_variable(input_df,imputation_strategies,standardize,my_variable_index):
     
#     variable_names=input_df.columns
#     N_variables=variable_names.shape[0]
#     available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     [values, keys]=dict2list(available_percentage_observations)
    
#     # #sort the variabls accoriding to decreasing or increasing number of observations
#     # argsort_idx=(-np.asarray(values)).argsort()
#     # #argsort_idx=(np.asarray(values)).argsort()
    
    
#     # #reorder input as decreasing number of observations
#     # input_df=input_df[variable_names[argsort_idx]]
#     # variable_names=input_df.columns
#     # N_variables=variable_names.shape[0]
#     # available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     # [values, keys]=dict2list(available_percentage_observations)

#     #my_df=(my_df-my_df.mean())/my_df.std()
#     #standardize=0
#     if standardize==1:
#         X_init=input_df.values
#         X_non_std=input_df.values

#         #change 999 by nan to can comoute means and stds easier...
#         input_df2=input_df.copy()
#         input_df2.replace(to_replace=[999],value=np.nan,inplace=True)
#         summary_stats=input_df2.describe().transpose() 
#         standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#         X_std=standard_df.values 
#         #put back the 999 coding
#         X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]

#         X_init=X_std

#     else:
#         X_init=input_df.values

        
#     X_init_back_up=X_init.copy()
#     X_filled=X_init.copy()
#     X_filled_back_up=X_filled.copy()

    
#     print('data matrix to input shape',X_init.shape)
#     n_samples, n_features = X_init.shape
#     #N_SPLITS = 100
#     #rng = np.random.RandomState(0)

#     #scores = pd.DataFrame()
#     #e=dict.fromkeys(['mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])


#     #imputation_strategies=['mean', 'median']#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression']
    
    
#     all_errors=dict.fromkeys(imputation_strategies)
#     all_X_filled=dict.fromkeys(imputation_strategies)
#     all_X_reimputing_good_values=dict.fromkeys(imputation_strategies)
    
#     #if standardize==1:
#     all_errors_original_std=dict.fromkeys(imputation_strategies)

    
#     for strategy in imputation_strategies: # ('mean', 'median'):#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'):
        
#         all_errors[strategy]=dict.fromkeys(pd.Index.tolist(variable_names))
#         all_errors_original_std[strategy]=dict.fromkeys(pd.Index.tolist(variable_names))

#         X_init=X_init_back_up.copy()
#         X_filled=X_filled_back_up.copy()
#         X_reimputing_good_values=999 * np.ones(X_init.shape)
        
#         #for col in range(N_variables): 
#         col=my_variable_index        
#         variable=variable_names[col]
#         #all_errors[variable]=dict.fromkeys(imputation_strategies)#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])

#         good_subjects = np.ndarray.flatten(np.argwhere(input_df[variable].values !=999))
#         missing_subjects=np.ndarray.flatten(np.argwhere(input_df[variable].values == 999))
#         #print('variable', variable, 'has', good_subjects.shape[0], 'observations (subjects)')
        
#         #substitute observed values by 999 for testing
#         for sub in range(good_subjects.shape[0]):
            
#             #if col ==0:
#             X_missing = X_init.copy()
#             #else:
#             #X_missing = X_filled.copy()
               
                
#             X_missing[good_subjects[sub], col] = 999 
            
#             inputed_data = my_inputer(X_missing,strategy)
                        
#             error=np.sqrt(np.power(X_filled[good_subjects[sub], col] - inputed_data[good_subjects[sub], col],2))
            
#             if standardize ==1:
#                 error_original=np.sqrt(np.power(X_non_std[good_subjects[sub], col] - np.add(np.multiply(inputed_data[good_subjects[sub], col],summary_stats['std'][col]),summary_stats['mean'][col]),2))
#                 #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        

            
#             tmp_value=inputed_data[good_subjects[sub], col]
            
#             #print(tmp_value)
            
#             #print(X_reimputing_good_values[good_subjects[sub], col])
            
#             X_reimputing_good_values[good_subjects[sub], col] = tmp_value
            
#             #print(X_reimputing_good_values[good_subjects[sub], col])

            
#             if sub==0:
#                 all_errors[strategy][variable]=error
#                 if standardize==1:
#                     all_errors_original_std[strategy][variable]=error_original
#             else:
#                 all_errors[strategy][variable]=np.append(all_errors[strategy][variable],error)
#                 if standardize==1:
#                     all_errors_original_std[strategy][variable]=np.append(all_errors_original_std[strategy][variable],error_original)
                
#         #impute the variable without adding dummy missing values
#         if col ==0:
#             X_missing = X_init.copy()
#         else:
#             X_missing = X_filled.copy()
               
#         inputed_data = my_inputer(X_missing,strategy)

#         X_filled[:,col] = inputed_data[:,col]
#         #END FOR

#         if standardize==0:
#             all_X_filled[strategy]=X_filled
#             all_X_reimputing_good_values[strategy]=X_reimputing_good_values
#         else:
#             all_X_filled[strategy]= np.multiply(X_filled + np.asarray(summary_stats['mean']),np.asarray(summary_stats['std']))
            
#             all_X_reimputing_good_values[strategy]=np.add(np.multiply(X_reimputing_good_values,np.asarray(summary_stats['std'])), np.asarray(summary_stats['mean']))
#             all_X_reimputing_good_values[strategy][np.where(X_init==999)]=999
#             #X_init[np.where(X_init==999)]
#             1
#             #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#             #X_std=standard_df.values 
#             ##put back the 999 coding
#             #X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]
#             #X_init=X_std
                
        
            
            
            

#     #scores= pd.DataFrame.from_dict(e)
#     initial_data = input_df.values
#     return all_errors, variable_names, all_X_filled, all_X_reimputing_good_values,initial_data, all_errors_original_std




# def imputations_comparison_one_variable_FINAL(input_df,strategy,standardize,my_variable_name,params):
     
#     variable_names=input_df.columns
#     N_variables=variable_names.shape[0]
#     available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     [values, keys]=dict2list(available_percentage_observations)
    
#     my_variable_index  =np.argwhere(variable_names==my_variable_name)[0][0]
    
#     # #sort the variabls accoriding to decreasing or increasing number of observations
#     # argsort_idx=(-np.asarray(values)).argsort()
#     # #argsort_idx=(np.asarray(values)).argsort()
    
    
#     # #reorder input as decreasing number of observations
#     # input_df=input_df[variable_names[argsort_idx]]
#     # variable_names=input_df.columns
#     # N_variables=variable_names.shape[0]
#     # available_observations,available_number_observations,available_percentage_observations = stats_on_missing_data(input_df)
#     # [values, keys]=dict2list(available_percentage_observations)

#     #my_df=(my_df-my_df.mean())/my_df.std()
#     #standardize=0
#     if standardize==1:
#         X_init=input_df.values
#         X_non_std=input_df.values

#         #change 999 by nan to can comoute means and stds easier...
#         input_df2=input_df.copy()
#         input_df2.replace(to_replace=[999],value=np.nan,inplace=True)
#         summary_stats=input_df2.describe().transpose() 
#         standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#         X_std=standard_df.values 
#         #put back the 999 coding
#         X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]

#         X_init=X_std

#     else:
#         X_init=input_df.values

        
#     X_init_back_up=X_init.copy()
#     X_filled=X_init.copy()
#     X_filled_back_up=X_filled.copy()

    
#     print('data matrix to input shape',X_init.shape)
#     n_samples, n_features = X_init.shape
#     #N_SPLITS = 100
#     #rng = np.random.RandomState(0)

#     #scores = pd.DataFrame()
#     #e=dict.fromkeys(['mean', 'median','Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])
#     col=my_variable_index        
#     variable=variable_names[col]

#     #imputation_strategies=['mean', 'median']#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression']
    
    
#     all_X_filled=dict.fromkeys([strategy])#imputation_strategies)
#     all_X_filled[strategy]=dict.fromkeys([my_variable_name])#imputation_strategies)

#     all_X_reimputing_good_values=dict.fromkeys([strategy])#imputation_strategies)
#     all_X_reimputing_good_values[strategy]=dict.fromkeys([my_variable_name])

    
    
#     #if standardize==1:
#     all_errors_original_std=dict.fromkeys([strategy])#imputation_strategies)
#     all_errors_original_std[strategy]=dict.fromkeys([variable])#dict.fromkeys(pd.Index.tolist(variable_names))

    
#     #for strategy in imputation_strategies: # ('mean', 'median'):#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'):   
#     all_errors=dict.fromkeys([strategy])#imputation_strategies)
#     all_errors[strategy]=dict.fromkeys([variable])#pd.Index.tolist(variable_names))



#     X_init=X_init_back_up.copy()
#     X_filled=X_filled_back_up.copy()
#     X_reimputing_good_values=999 * np.ones(X_init.shape)
    
#     #for col in range(N_variables): 
    
#     #all_errors[variable]=dict.fromkeys(imputation_strategies)#,'Bayesian_Ridge_Regression','Decision_Tree_Regression','Extra_Trees_Regressor','Kneighbours_Regression'])

#     good_subjects = np.ndarray.flatten(np.argwhere(input_df[variable].values !=999))
#     missing_subjects=np.ndarray.flatten(np.argwhere(input_df[variable].values == 999))
#     #print('variable', variable, 'has', good_subjects.shape[0], 'observations (subjects)')
    
#     #substitute observed values by 999 for testing
#     for sub in range(good_subjects.shape[0]):
        
#         #if col ==0:
#         X_missing = X_init.copy()
#         #else:
#         #X_missing = X_filled.copy()
           
            
#         X_missing[good_subjects[sub], col] = 999 
        
#         inputed_data = my_inputer_FINAL(X_missing,strategy,params)
                    
#         error=np.sqrt(np.power(X_filled[good_subjects[sub], col] - inputed_data[good_subjects[sub], col],2))
        
#         if standardize ==1:
#             error_original=np.sqrt(np.power(X_non_std[good_subjects[sub], col] - np.add(np.multiply(inputed_data[good_subjects[sub], col],summary_stats['std'][col]),summary_stats['mean'][col]),2))
#             #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        

        
#         tmp_value=inputed_data[good_subjects[sub], col]
        
#         #print(tmp_value)
        
#         #print(X_reimputing_good_values[good_subjects[sub], col])
        
#         X_reimputing_good_values[good_subjects[sub], col] = tmp_value
        
#         #print(X_reimputing_good_values[good_subjects[sub], col])

        
#         if sub==0:
#             all_errors[strategy][variable]=error
#             if standardize==1:
#                 all_errors_original_std[strategy][variable]=error_original
#         else:
#             all_errors[strategy][variable]=np.append(all_errors[strategy][variable],error)
#             if standardize==1:
#                 all_errors_original_std[strategy][variable]=np.append(all_errors_original_std[strategy][variable],error_original)
            
#         #impute the variable without adding dummy missing values
#         if col ==0:
#             X_missing = X_init.copy()
#         else:
#             X_missing = X_filled.copy()
               
#         inputed_data = my_inputer(X_missing,strategy)

#         X_filled[:,col] = inputed_data[:,col]
#         #END FOR

#         if standardize==0:
#             all_X_filled[strategy][my_variable_name]=X_filled[:,my_variable_index]
#             all_X_reimputing_good_values[strategy][my_variable_name]=X_reimputing_good_values[:,my_variable_index]
#         else:
#             all_X_filled[strategy]= np.multiply(X_filled + np.asarray(summary_stats['mean']),np.asarray(summary_stats['std']))
            
#             all_X_reimputing_good_values[strategy]=np.add(np.multiply(X_reimputing_good_values,np.asarray(summary_stats['std'])), np.asarray(summary_stats['mean']))
#             all_X_reimputing_good_values[strategy][np.where(X_init==999)]=999
#             #X_init[np.where(X_init==999)]
#             1
#             #standard_df= (input_df2 -summary_stats['mean'])/summary_stats['std']        
#             #X_std=standard_df.values 
#             ##put back the 999 coding
#             #X_std[np.where(np.isnan(standard_df)==True)]=X_init[np.where(np.isnan(standard_df)==True)]
#             #X_init=X_std
                
        
            
            
            

#     #scores= pd.DataFrame.from_dict(e)
#     initial_data = input_df[my_variable_name].values
#     return all_errors, variable_names, all_X_filled, all_X_reimputing_good_values,initial_data, all_errors_original_std