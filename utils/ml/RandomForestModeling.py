import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from .RandomForestAlgorithm import RandomForest


# function to train RF entirely
def RandomForest_modeling(X, y, kfold, maxDepths, maxFeatures, numTrees, criterions, variant='all'):

    # split materials data into tranining and test set
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, stratify=y, test_size=0.20)

    if variant == 'all':
        n = 4
        n_variant = [None, 'ImbalanceCost', 'FixedRatio','Undesampling']
    else:
        n = 1
        n_variant = [variant]
    
    opt_models = []    
    trn_A = np.zeros((n,len(maxDepths),len(maxFeatures)))
    val_A = np.zeros((n,len(maxDepths),len(maxFeatures)))

    # for each maxDepths
    for i, maxDepth in enumerate(maxDepths):

        # foe each maxFeatures
        for j, maxFeature in enumerate(maxFeatures):        

            # k-fold validation
            skf = StratifiedKFold(n_splits=kfold)
            for lrn_index, val_index in skf.split(X_trn, y_trn):
                
                X_lrn, X_val = X_trn[lrn_index], X_trn[val_index]
                y_lrn, y_val = y_trn[lrn_index], y_trn[val_index]

                ## perform RF
                for k in range(n):
                    trnA, valA, _ = RandomForest_train(X_lrn, y_lrn, 
                                                       X_val, y_val, 
                                                       numTrees, 
                                                       maxFeature,
                                                       maxDepth=maxDepth,
                                                       classWeight=n_variant[k])
                    trn_A[k,i,j] += trnA
                    val_A[k,i,j] += valA

    # averaging training and validation accuracy
    trn_A /= kfold
    val_A /= kfold

    # calculate class imbalance weight 
    weight = {1 : sum(y_trn==-1)/sum(y_trn==1)}
    
    # perform model testing
    for k in range(n):
        
        idx = np.where(val_A[k] == val_A[k].max())
        opt_maxDepth, opt_maxFeat = maxDepths[idx[0][0]], maxFeatures[idx[1][0]]
        trn_A, tst_A, opt_model = RandomForest_train(X_trn, 
                                                     y_trn, 
                                                     X_tst, 
                                                     y_tst, 
                                                     numTrees, 
                                                     opt_maxFeat,
                                                     maxDepth=opt_maxDepth, 
                                                     classWeight=n_variant[k], 
                                                     weights=weight)
        
        print('%s RF \n opt_maxDepth: %d \n opt_maxFeat: %d \n Train_A: %f \n Test_A: %f\n' % 
              (n_variant[k], opt_maxDepth, opt_maxFeat, trn_A, tst_A))
        
        opt_models.append(opt_model)
    

    return opt_models


# function to train RF
def RandomForest_train(X_trn, y_trn, X_tst, y_tst, numTree, maxFeature, maxDepth=3, 
             criterion='gini', classWeight=None, weights=None):
    
    model = RandomForest(numTree=numTree, 
                         maxFeature=maxFeature, 
                         maxDepth=maxDepth, 
                         criterion=criterion, 
                         classWeight=classWeight)     
    model.fit(X_trn,y_trn)
    
    ypred = model.predict(X_trn)
    trn_A = model.accuracy(y_trn, ypred, weights)
    
    ypred = model.predict(X_tst)
    tst_A = model.accuracy(y_tst, ypred, weights)
    
    return trn_A, tst_A, model



def extract_feature_importance(names, model):
    
    idx = np.argsort(-model.feature_importance())
    feature_names = names[idx]
    feature_importances = -np.sort(-model.feature_importance())
    
    return idx, feature_names, feature_importances

    
    
