import pandas as pd
import numpy as np
from .estatisticas import *
import time
from datetime import datetime
from .preprocessing import *

'''
Algoritmos de treino e de teste 
'''

def test_carregamento():
    print('teste 2')


class ModelCreate():
    def __init__(self, model, pre_process = None, filter_water=True, wrfoutput = None ):
        self.model = model
        self.pre_process_base = pre_process
        self.filter_water = filter_water
        self.train_metrics = pd.DataFrame()
        
        if wrfoutput is not None:
            self.database = DataBase(wrfoutput)
            self.mask_lulc = np.array(self.database.lulc==17)
            print('Base carregada')
        
    # create grade:
    @staticmethod
    def GradeCreatorDom_f(Y,shape = (85,195)):
        '''
        cria a grade no domínio do shape l,c
          
        Y: array com dados
        shape = dimensoes (l,c) l - linhas, c - colunas
        '''
        l = shape[0]
        c = shape[1]
        t2 = Y.reshape(l,c)
        return t2
    
    def _dataframe_transform(self,df):
        aux = pd.DataFrame()
        for i in df.keys():
            aux[i] = df[i].flatten()
        return aux
    
    def predict(self,x_test):
        """
        Return y_pred values to model
        -------------------------------------
        Input: pd.Dataframe
        
            x_test: values x to predict
        output:
            pd.DataFrame whith MSE, RMSE, r, r2 and NS
        """  
                #x_test = self.preprocesss_step(x_test)

        
        # aplicando o modelo
        shp = x_test[list(x_test.keys())[0]].shape
        
        x_test = self._dataframe_transform(x_test)
        
        
        #retirando pontos com agua
#         if self.filter_water:
#             #x_test['id']  = x_test.index
#             idW = x_test.loc[self.mask_lulc].index
#             idNW = x_test.loc[~self.mask_lulc].index
#             x_test = x_test.loc[~self.mask_lulc]
        

        y_pred = self.model.predict(x_test)
        # criando saida 
#         if self.filter_water:
#             df_aux = self._mask_water_apply(idNW,idW,'VAR',y_pred)
#             y_pred = df_aux['VAR'].values
        try:
            y_pred = y_pred.reshape(shp[0],shp[1])
            print('campo reorganizado')
        except:
            print('Vetor unico')
        return y_pred
        
    def _mask_water_apply(self, idNW,idW,var_name,var_data):
        df_agua = pd.DataFrame({'id':idW, var_name: np.full([idW.shape[0]], np.nan)})
        df_sem_agua = pd.DataFrame({'id':idNW, var_name: var_data})
        df_aux = df_agua.append(df_sem_agua)
        df_aux = df_aux.sort_values(['id'])
        
        return df_aux[var_name]
        
    def metricas(self,s,o):        
        """
        Return MSE, RMSE, r, r2 and NS
        nput:
            s: simulated
            o: observed
        output:
            MSE, RMSE, r, r2 and NS
        """  
        Mae = mae(s,o)
        Mse = mse(s,o)
        Rmse = rmse(s,o)
        R_coef = r(s,o)
        R_2   = r2(s,o)
        NSE = NS(s,o)

        return [Mae,Mse, Rmse, R_coef, R_2, NSE]

    def preprocesss_step(self,df):
        if self.pre_process_base is not None:
            df = self.pre_process_base.fit_transform(df)
            return df
        else:
            return df
        
    
    def fit(self, x, y,index_name='Ensaio'):
        ti = time.time()
        
#       # Dataframe tranform
        x_train = self._dataframe_transform(x)
        
        y=y.flatten()

        
        if self.filter_water:
            x_train['Y'] = y
            y = x_train['Y'][~self.mask_lulc]
            x_train = x_train[~self.mask_lulc]
            x_train.drop('Y',axis=1,inplace=True)
        
        # Treinando o modelo
        self.model.fit(x_train,y)
        y_train = self.model.predict(x_train)
        self.x_train = x_train
        self.train_metrics = pd.concat([self.train_metrics,errcoef(y,y_train,index_name)])
        
        
         
    
    
    def fit_batch(self,X_base,x_test,delta, target):
        inicio = datetime.now()
        
        self.target_base = self.database.get_var_din(target,range=delta)        
        
        lista = []
        for i in range(self.target_base.shape[0]):
            print(i) 
            print(self.target_base.shape)
            
            y_train = self.target_base[i].flatten()
            print(f'shape y: {y_train.shape[0]}')
            #assert X_base[list(X_base.keys())[0]].flatten().shape[0]==y_train.shape[0]
            
            self.fit(X_base,y_train,index_name='time_pass_'+str(i))
            y_pred = self.predict(x_test)
            lista.append(y_pred)
            
        self.test_result = np.array(lista)
        fim = datetime.now()
        print(f'tempo de processamento: {fim - inicio}')
    
    @staticmethod       
    def get_icu(lclu, T, shape=(85,195)):
        """
        Return Urban Heat island from T domain 
        
        -------------------------------------------
        inputs:
            lclu: pd.Series, Array(1D or 2D), list
            land canopy and land use where int 13 is urban use.
            
        outputs:array 2
            maps with Urban Heat Island
            
        """
        if isinstance(lclu, pd.DataFrame):
            lclu = lclu.values
        else:
            lclu= np.array(lclu)
        
        T = T.flatten()
        t_rural_mean = T[(lclu!=13) & (lclu!=17)].mean()
        T_icu = T - t_rural_mean
        
        return ModelCreate.GradeCreatorDom_f(T_icu, shape=shape)
            
            
            