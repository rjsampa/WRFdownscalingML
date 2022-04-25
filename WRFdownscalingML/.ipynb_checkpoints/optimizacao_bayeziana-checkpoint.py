
import xgboost as xgb
from bayes_opt import BayesianOptimization
import pandas as pd

from datetime import datetime
        
class optimizeXGBClassifierBO():   
    def __init__(self,x_train, x_test,y_train,y_test, verbose=0):
        '''
    Classe para optmizacao do XGB utilizando optimizacao Bayesiana[1]. Tem por base o XGBClassifier. 
    
    inputs:
        x_train: pd.Dataframe - Dataframe com as features para treino
        x_test: pd.Dataframe - Dataframe com as features para teste
        y_train: pd.Dataframe - Dataframe com o target para treino
        y_test: pd.Dataframe - Dataframe com o target para teste
    
    methods:
        fit: executa a optmizacao
        XGB_BO: class - BayesianOptimization class
        params: dict - parametros do melhor modelo
        estatistica_treino: tuple - metricas (KS, Gini, AUC) do melhor modelo para treino 
        estatistica_teste: tuple -  metricas (KS, Gini, AUC) do melhor modelo para teste 
        
    
    -------------------------------------------------------------------------------------------------
    Exemplo:
    
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state=42)
        optimizador = optimizeXGBClassifierBO(X_train,X_test,y_train,y_test)
        optimizador.fit(n_iter=30)
        resultados = pd.DataFrame(optimizador.XGB_BO.res).sort_values('target',ascending=False)

    
    
        
    1 - https://github.com/fmfn/BayesianOptimization       
    ''' 
        self.dtrain = xgb.DMatrix(x_train, label=y_train)
        del(x_train)
        self.dtest = xgb.DMatrix(x_test)
        del(x_test)
        self.y_train = y_train
        self.y_test = y_test
        self.verbose = verbose
        
        
    def xgb_evaluate(self,max_depth,gamma,colsample_bytree,min_child_weight,max_delta_step,subsample,eta,learning_rate,scale_pos_weight,reg_alpha,reg_lambda):
        T1 = datetime.now()
        params = {'eval_metric': 'rmse',
                   'objective' : 'reg:squarederror',
                  'nthread' : 4,
                  'max_depth': int(max_depth),
                  #'subsample' : max(min(subsample, 1), 0),
                  'subsample': subsample,
                  'eta': eta,
                  'gamma': gamma,
                  'min_child_weight' : min_child_weight,
                  'max_delta_step' : int(max_delta_step),
                  'colsample_bytree': colsample_bytree,
                  'learning_rate':learning_rate,
                  'scale_pos_weight':scale_pos_weight,
                   'reg_alpha':reg_alpha,
                   'reg_lambda':reg_lambda
                   
                  }
        # Used around 1000 boosting rounds in the full model
        cv_result = xgb.cv(params, 
                           self.dtrain, 
                           num_boost_round=100, 
                           metrics='rmse', 
                           nfold=3)

        T2 = datetime.now()

        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0*cv_result['test-rmse-mean'].iloc[-1]
    
    def fit(self,n_iter=5, params={'max_depth': (5, 9),
                                   'gamma': (0.001, 1),
                                   'min_child_weight': (5, 20),
                                   'scale_pos_weight': (1.2, 5),
                                   'reg_alpha': (4.0, 7.0),
                                   'reg_lambda': (1.0, 7.0),
                                   'max_delta_step': (1, 5),
                                   'subsample': (0.4, 1.0),
                                   'colsample_bytree': (0.3, 1.0),
                                   'eta': (0.1, 1.0),
                                   'learning_rate': (0.1, 1.0)}, return_metrics=False):
     
        
        """
        Realiza optimizacao
        input:
            n_iter: int - Numero de interacoes 
        """
        
        
#         {'max_depth': (2, 7), # Máximo 7 para implantação. Não ultrapassar 7!
#                                    'gamma': (0.001, 10.0),
#                                    'min_child_weight': (0, 20),
#                                    'max_delta_step': (0, 10),
#                                    'subsample': (0.4, 1.0),
#                                    'colsample_bytree':(0.4, 1.0),
#                                   }
        
        
        self.XGB_BO = BayesianOptimization(self.xgb_evaluate, 
                                           params, verbose=self.verbose
                                          )
        
        self.XGB_BO.maximize(init_points=3,n_iter=n_iter,acq='ei')
        
        self.params = pd.DataFrame(self.XGB_BO.res).sort_values('target', ascending = False).iloc[0,1]
        self.params['max_depth'] = int(self.params['max_depth'])
        
#         print('-'*130)
#         print('Final Results')
#         print('Best XGBOOST parameters: ', self.params)
        
        
        # Train a new model with the best parameters from the search
        #model2 = xgb.train(self.params, self.dtrain, num_boost_round=250)

        # Predict on testing and training set
        #y_pred_test = model2.predict(self.dtest)
        #y_pred_train = model2.predict(self.dtrain)

        
        

        
#--------------------------------------------------------------------------------------------------------------
# Baseado em https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm

import lightgbm as lgb
from bayes_opt import BayesianOptimization
import numpy as np

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=0, n_estimators=10000, learning_rate=0.05):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters

    def lgb_eval(num_leaves, feature_fraction, max_depth , min_split_gain, min_child_weight):
        params = {
            "objective" : "regression", "bagging_fraction" : 0.8, "bagging_freq": 1,
            "min_child_samples": 20, "reg_alpha": 1, "reg_lambda": 1,"boosting": "rf",
            "learning_rate" : 0.01, "subsample" : 0.8, "colsample_bytree" : 0.8, "verbosity": -1, "metric" : 'rmse'
        }
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval =200,stratified=False)
        return (-1.0 * np.array(cv_result['rmse-mean'])).max()
    
        # range 
    lgbBO = BayesianOptimization(lgb_eval, {'feature_fraction': (0.1, 0.9),
                                            'max_depth': (5, 9),
                                            'num_leaves' : (200,300),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0,verbose=0)
        # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round,acq='ei')
    params = pd.DataFrame(lgbBO.res).sort_values('target', ascending = False).iloc[0,1]
    params['max_depth'] = int(params['max_depth'])
    params['num_leaves'] = int(params['num_leaves'])

    return params#.res['max']['max_params']

"""
{'max_depth': (2, 7), # Máximo 7 para implantação. Não ultrapassar 7!
                                   'gamma': (0.001, 10.0),
                                   'min_child_weight': (0, 20),
                                   'max_delta_step': (0, 10),
                                   'subsample': (0.4, 1.0),
                                   'colsample_bytree':(0.4, 1.0),
                                   'eta': (0.1, 1.0)}
"""
# Random forest 

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#General function for building a model
def rf_cv(X, y, **kwargs):
    estimator = RandomForestRegressor(**kwargs)
    cval = cross_val_score(estimator, X, y, scoring = 'neg_mean_squared_error', cv = 4, verbose = 0, n_jobs = -1)
    return cval.mean()


def bayesian_optimise_rf(X, y, n_iter = 100):
    def rf_crossval(n_estimators, max_features):
        #Wrapper of RandomForest cross validation.
        #Note the fixing of the inputs so they match the expected type
        #(e.g n_estimators must be an integer)
        return rf_cv(
            X = X,
            y = y,
            n_estimators = int(n_estimators),
            max_features = max(min(max_features, 0.999), 1e-3),
            bootstrap = True
        )
    
    optimizer = BayesianOptimization(verbose=0,
        f = rf_crossval,
        pbounds = {
            "n_estimators" : (10, 400),
            "max_features" : (0.1, 0.999),
        }
    )
    optimizer.maximize(n_iter = n_iter)
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    return best_params