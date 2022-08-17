import numpy as np
import pandas as pd

import sys
sys.path.append("..\\..")

from utils.ml.RandomForestModeling import RandomForest_modeling
from utils.ml.RandomForestModeling import extract_feature_importance
from utils.visualization.visualization import hist_predicted_output

def global_modeling():
    
    # import data
    data_path = '..\\..\\data\\vdW_nvdW\\preprocessed_materials_project_data.csv'
    data = pd.read_csv(data_path).drop(columns=['Unnamed: 0'])
    
    # get input X and output y
    feature_names = np.array(['density', 'formation_energy_per_atom','volume','band_gap','max_abc'])
    X = np.array(data[feature_names])
    y = np.array(data['Dim'])

    
    print('Size of dataset: ', data.shape)
    print('Number of vdW: %d, Number of nvdW: %d' % (sum(y==1), sum(y==-1)))
    print('Imbalance Ratio: ', sum(y==-1)/sum(y==1))
    
    # Hyperparameter setup
    numTrees = 200
    maxFeatures = [2,3,4]
    maxDepths= [7,9,11,13,15]
    criterions='gini'
    kfold = 10
    variant = 'ImbalanceCost'
    
    print('Begin Modeling...', end='')
    model_global = RandomForest_modeling(X=X, 
                                         y=y, 
                                         kfold=kfold, 
                                         maxDepths=maxDepths, 
                                         maxFeatures=maxFeatures, 
                                         numTrees=numTrees, 
                                         criterions=criterions, 
                                         variant=variant)
    print('done!')
    
    
    ## extract features importance
    idx, fi_names, fi_values = extract_feature_importance(feature_names, model_global[0])
    print('Feature Importance:')
    print(fi_names)
    print(fi_values)
    
    
    ## generate histogram of predicted output figure
    tickxloc = np.array([[2,6,10,14,18],[-4,-3,-2,-1,0],[500,1500,2500,3500,4000],[0,2,4,6,8],[25,50,75,100,125]])
    class_name = ['vdW', 'nvdW']
    winning_features = feature_names[idx][:2]
    winning_tickxloc = tickxloc[idx][:2]
    
    
    ## generate predicted output
    ypred = model_global[0].predict(X)
    HOPO_data = np.concatenate((X[:,idx], ypred.reshape(-1,1)), axis=1)


    ## plot histogram    
    hist_predicted_output(HOPO_data, 
                          winning_features,
                          class_name,
                          density=True, 
                          tickxloc=winning_tickxloc)    
    
    
    
if __name__ == "__main__":
    
    global_modeling()