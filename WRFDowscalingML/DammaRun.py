import pandas as pd 
import numpy as np 
from .preprocessing import *
from .Train import *


class Damma():
    '''
    Downscalling de campos oriundos do wrf 
    '''
    def fit(self,VAR, d1 , d2 ,init , end, regressor):
        '''
        Realiza o downscaling estatistico dos dados de saida do wrf 
        
        input:
            VAR: Váriavel de interesse no arquivo netCDF do WRF
            d1: output do wrf referente ao dom inicial
            d2: output do wrf referente ao dom final na escala espacial que se deseja realizar o downscaling
            init: passo inicial tendo por base os arquivos netcdf
            end: passo final
            regressor: estimador a ser utilizado, estruturado de forma "model.fit"
            
        '''
        # abrindo  nc
        dt1 = dataset(d1)
        dt2 = dataset(d2)
        
        # separando variaveis Di
        
        mde1 = dt1.mde()
        self.mdeD1 = mde1.flatten()
        self.lulcD1 = dt1.landUse().flatten()
        self.latD1 = dt1.xlat().flatten()
        self.lonD1 = dt1.xlong().flatten()
        slopeD1 , aspectD1 = slope_aspect(mde1)
        self.slopeD1 = slopeD1.flatten()
        self.aspectD1 = aspectD1.flatten()
        
        # criando DataFrame Di
        lista = [self.mdeD1,self.lulcD1,self.latD1,self.lonD1,self.slopeD1,self.aspectD1]
        nomes = [ 'topo', 'lu_index','xlat','xlong', 'slope', 'aspect']
        domiX = pandasDF(lista,nomes)
        
        t2md1 = dt1.getVar(VAR)
        self.varD1 = t2md1[init : end]
        
        # separando variaveis Df
        mde2 = dt2.mde()
        self.mdeD2 = mde2.flatten()
        self.lulcD2 = dt2.landUse().flatten()
        self.latD2 = dt2.xlat().flatten()
        self.lonD2 = dt2.xlong().flatten()        
        slopeD2 , aspectD2 = slope_aspect(mde2)
        self.slopeD2 = slopeD2.flatten()
        self.aspectD2 = aspectD2.flatten()
        
        # criando DataFrame Df
        lista = [self.mdeD2,self.lulcD2,self.latD2,self.lonD2,self.slopeD2,self.aspectD2]
        nomes = [ 'topo', 'lu_index','xlat','xlong', 'slope', 'aspect']
        domfX = pandasDF(lista,nomes)
        
        t2md2 = dt2.getVar(VAR)
        self.varD2 = t2md2[init : end]
        
        # resolvendo conflitos de classes de uso do solo
        domiX.lu_index[domiX['lu_index']==6]=7
        domiX.lu_index[domiX['lu_index']==16]=10
        
        domfX.lu_index[domfX['lu_index']==6]=7
        domfX.lu_index[domfX['lu_index']==16]=10
        
        

        # aplicando downscaling
        # Padroniza
        domiX = X_structure(domiX)
        domfX = X_structure(domfX)
        self.dom_i = PreProcessingX(domiX)
        self.dom_f = PreProcessingX(domfX)
        
        
        self.Ypred , self.regressor = train(self.varD1, self.dom_i, self.dom_f, model = 3, regressor=regressor)
        
        
    

        
        
        
