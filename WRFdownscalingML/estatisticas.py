import pandas as pd
import numpy as np

'''
Fornece ferramentas estatisticas para avaliar o desempenho dos modelos, tendo como base dados observados e modelados
'''
# funcoes
def remove_nan(o,s):
    '''
    y o valor '''
    o = np.array(o)
    s = np.array(s)
    o = o[~np.isnan(s)]
    s = s[~np.isnan(s)]
    return s, o

def rmse(s,o):    
    """
    Raiz quadrada do erro medio 
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s,o = remove_nan(o,s)
    
    return np.sqrt(np.mean((s-o)**2))

def mae(s,o):
    """
    Erro medio absoluto
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    s,o = remove_nan(o,s)
    return np.mean(abs(s-o))

def mse(s,o):
    """
    Erro quadratico medio 
    input:
        s: simulated
        o: observed
    output:
        maes: mean squared error
    """
    s,o = remove_nan(o,s)
    return np.mean((s-o)**2)

def r2(s,o):
    """ 
    Coeficiente de determinacao
    input: 
        s: simulated
        o: observed
    output:
        r2: Coefficient of determination
        """
    s,o = remove_nan(o,s)
    return (np.corrcoef(s,o)[0,1])**2

def r(s,o):
    """ 
    coeficiente de correlacao
    input: 
        s: simulated
        o: observed
    output:
        r: Coefficient of correlation
    """
    s,o = remove_nan(o,s)
    return np.corrcoef(o,s)[0,1]

def mape(s,o):
    s,o = remove_nan(o,s)
    mape_v = abs((abs(o-s).sum()/o.sum()))
    
    return mape_v


def accuracy(mape_v):
    return 1 - mape_v
    

def NS(s,o):
    """
    coeficiente de eficiência do modelo de Nash
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """   
    s,o = remove_nan(o,s)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

# main function
def errcoef(s,o,nome='Teste'):
    """
    Return MSE, RMSE, r, r2 and NS
    nput:
        s: simulated
        o: observed
    output:
        MSE, RMSE, r, r2 and NS
    """     
    MAPE = mape(s,o) 
    Facc = accuracy(MAPE)
    Mae = mae(s,o)
    Mse = mse(s,o)
    Rmse = rmse(s,o)
    R_coef = r(s,o)
    R_2   = r2(s,o)
    NSE = NS(s,o)
    #Mae,Mse, Rmse, R_coef, R_2, NSE
    df = pd.DataFrame({'MAE':Mae,'MSE':Mse,
                       'RMSE':Rmse,'R':R_coef,
                       'R2':R_2, 'NSE':NSE,
                       'MAPE':MAPE,'Facc':Facc}, index = [nome])
    
    return df


    
def avaliando(s,o):
    """ 
    retorna um pandas dataframe com coeficientes de erro
    input:
        s: dataframe com valores simulados
        o: dataframe com valores observados
    output: 
        df: dataframe com os erros 
    """
    estacoes = s.columns
    df = pd.DataFrame()
        

    for i in estacoes:
        w = errcoef(s[i].values, o[i].values,i)
        df = df.append(w)
    
    return df
        
def coefsplit(lista, nomes):
    nse = pd.DataFrame()
    mse = pd.DataFrame()
    rmse = pd.DataFrame()
    r = pd.DataFrame()
    r2 = pd.DataFrame()
    for k , i in enumerate(lista):
        nse[nomes[k]] = i['NSE']
        mse[nomes[k]] = i['MSE']
        rmse[nomes[k]] = i['RMSE']
        r[nomes[k]] = i['R']
        r2[nomes[k]] = i['R2']
       
    
    return nse, mse, rmse, r, r2