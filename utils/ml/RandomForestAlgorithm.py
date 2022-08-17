import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    
    def __init__(self, numTree, maxFeature, maxDepth=3, criterion='gini', classWeight=None):
        """ Random Forest ALgorithm focusing on imbalance class.
        
        Input parameters:
            numTree [int]: number of decision tree
            
            maxFeatures [int]: maximum number of features selected in each tree
            
            maxDepth [int]: maximum tree depth for each tree (default: 3)
            
            criterion [string]: criterion of decision tree [gini, entropy] (default: gini)
            
            classWeight [string]: type of approach to deal with class imbalance (default: None)
            
                - None: Standard RF with no imbalance technique implemented.
                
                - ImbalanceCost: For each tree, compute the imbalance ratio R of the bootstrapped samples, 
                                 then alter the class weight  
                                 
                - FixedRatio: Fix sample class ratio during the bootstraping step, the imbalance ratio R 
                              will always be the same for every tree
                              
                - Undersampling: During bootstraping step, perform undersampling on the majority samples        
                
        """      
        
        self.numTree = numTree
        self.maxFeature = maxFeature
        self.maxDepth = maxDepth
        self.criterion = criterion
        self.classWeight = classWeight
        
        
    def fit(self, X, y):
        """Perform model training."""
        
        ensemble_tree = []
        
        # calculate class weight for accuracy calculation
        self.weight = self.weight_calculation(X, y)        
        
        for nt in range(self.numTree):
            
            # obtain bootstrapped samples
            X_bootstrap, y_bootstrap = self.bootstrap(X, y)
            
            # randomly select features up to maxFeatures
            X_bootstrap, idx_feature = self.max_feature(X_bootstrap)
            
            # obtain class weight 
            weight = self.weight_calculation(X_bootstrap, y_bootstrap)
            
            # perform decision tree
            model = DecisionTreeClassifier(class_weight=weight, criterion=self.criterion, 
                                           max_depth=self.maxDepth)
            model.fit(X_bootstrap,y_bootstrap)
            
            # Save the decision tree and the selected features
            ensemble_tree.append([model,idx_feature])
            
        # save model
        self.model = ensemble_tree
        self.inputDim = X.shape[1]
    
    
    def predict(self, X):
        """Perform prediction on input samples."""
        
        # to make sure RF model is trained
        assert len(self.model) > 0
        
        n_sample, n_feature = X.shape
        ypred = np.zeros((n_sample))
        
        # for each tree,
        for model, idx_feature in self.model:
            #remove unwanted features for each tree
            X_feat = X[:,idx_feature]
            # sum all predicted value of each tree
            ypred += model.predict(X_feat)
        
        # average ypred by number of tree, and classify as 1 if ypred >= 0 , else 0
        ypred = ypred/self.numTree
        ypred[ypred>=0] = 1
        ypred[ypred<0] = -1
        
        return ypred  
    
            
    def weight_calculation(self, X, y):
        """ implement imbalance ratio R for imbalanceCost and FixedRatio cases """
        
        if self.classWeight == 'ImbalanceCost' or self.classWeight == 'FixedRatio':
            cls = np.unique(y)
            cls0_size = sum(y==cls[0])
            cls1_size = sum(y==cls[1])

            if cls0_size > cls1_size:
                weight = {cls[1] : cls0_size/cls1_size}
            else:
                weight = {cls[0] : cls1_size/cls0_size}

        else:
            weight = None
            
        return weight
    
    
    def max_feature(self, X):
        """Randomly select maxFeature number of feature"""
    
        idx_feature = np.random.choice(X.shape[1], self.maxFeature, replace=False)
        
        return X[:,idx_feature], idx_feature
    
            
    def bootstrap(self, X, y):
        '''Perform bootstrapping step based on the type of class weight'''

        # perform bootstraping based on classweight type
        if self.classWeight == None or self.classWeight == 'ImbalanceCost':
            X_bootstrap, y_bootstrap = self._ImbalanceCost(X,y)

        elif self.classWeight == 'FixedRatio':
            X_bootstrap, y_bootstrap = self._FixedRatio(X,y)

        elif self.classWeight == 'Undersampling':
            X_bootstrap, y_bootstrap = self._Undersampling(X,y)
            
        return X_bootstrap, y_bootstrap

    
    def feature_importance(self):
        """ Obtain feature importance of each features"""
        
        Dim = self.inputDim
        feature_importance = np.zeros((Dim,))
        
        for model, idx_feature in self.model:        
            feature_importance[idx_feature] += model.feature_importances_
        
        feature_importance /= len(self.model)
    
        return feature_importance / np.sum(feature_importance)
    
                                                       
    def _ImbalanceCost(self, X, y):
        """Imbalance Cost bootstrapping step"""
        
        idx_bootstrap = np.random.choice(X.shape[0],X.shape[0])
        X_bootstrap = X[idx_bootstrap,:]
        y_bootstrap = y[idx_bootstrap]
                                                       
        return X_bootstrap, y_bootstrap
                                                       
    
    def _FixedRatio(self, X, y):
        """Fix imbalance ratio during bootstrapping step"""
        
        # separate samples based on class
        cls = np.unique(y)
        X_cls0 = X[y==cls[0],:]
        y_cls0 = y[y==cls[0]]
        X_cls1 = X[y==cls[1],:]
        y_cls1 = y[y==cls[1]]

        # bootstrap samples from both classes
        idx_bootstrap_cls0 = np.random.choice(X_cls0.shape[0],X_cls0.shape[0])
        idx_bootstrap_cls1 = np.random.choice(X_cls1.shape[0],X_cls1.shape[0])
        X_bootstrap_cls0 = X_cls0[idx_bootstrap_cls0,:]
        y_bootstrap_cls0 = y_cls0[idx_bootstrap_cls0]
        X_bootstrap_cls1 = X_cls1[idx_bootstrap_cls1,:]
        y_bootstrap_cls1 = y_cls1[idx_bootstrap_cls1]

        # combine samples from both classes together
        X_bootstrap = np.concatenate((X_bootstrap_cls0, X_bootstrap_cls1), axis=0)
        y_bootstrap = np.concatenate((y_bootstrap_cls0, y_bootstrap_cls1), axis=0)       

        return X_bootstrap, y_bootstrap
    
    
    def _Undersampling(self, X, y):
        """Perform undersampling during bootstraping step"""
        
        cls = np.unique(y)
        X_cls0 = X[y==cls[0],:]
        y_cls0 = y[y==cls[0]]
        X_cls1 = X[y==cls[1],:]
        y_cls1 = y[y==cls[1]]

        # find smaller class sample size
        if X_cls0.shape[0] < X_cls1.shape[0]:
            size = X_cls0.shape[0]
        else:
            size = X_cls1.shape[0]

        # bootstrap samples from both classes (undersampling step included)
        idx_bootstrap_cls0 = np.random.choice(X_cls0.shape[0],size)
        idx_bootstrap_cls1 = np.random.choice(X_cls1.shape[0],size)
        X_bootstrap_cls0 = X_cls0[idx_bootstrap_cls0,:]
        y_bootstrap_cls0 = y_cls0[idx_bootstrap_cls0]
        X_bootstrap_cls1 = X_cls1[idx_bootstrap_cls1,:]
        y_bootstrap_cls1 = y_cls1[idx_bootstrap_cls1]

        # combine samples from both classes together
        X_bootstrap = np.concatenate((X_bootstrap_cls0, X_bootstrap_cls1), axis=0)
        y_bootstrap = np.concatenate((y_bootstrap_cls0, y_bootstrap_cls1), axis=0)  
    
        return X_bootstrap, y_bootstrap
    
    
    def accuracy(self, ytrue, ypred, weights=None):
        """Calculate normalized accuracy
        Input:
        ytrue [n,]     : n array containing true y
        ypred [n,]     : n array containing predicted y
        weights [dict] : dict containing the specified class weight. If None, 
                         the weight will be extracted from the trained RF classs.
        
        Output:
        accuracy [int]: accuracy value
        """
        
        cls = np.unique(ytrue)     
        if weights == None:
            if self.weight == None:
                weight = {cls[0] : 1}
            else:
                weight = self.weight
        else:
            weight = weights
            
        R = list(weight.values())[0]
            
        tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
        
        if [*weight] == cls[0]:  # if key of weight is equal to class 0
            error = (R*fp + fn) / ((fn + tp) + R*(tn + fp))
        else: 
            error = (fp + R*fn) / (R*(fn + tp) + (tn + fp))
        
        return 1 - error
    

    