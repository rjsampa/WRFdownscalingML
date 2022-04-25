import pandas as pd 
import numpy as np

'''
Manipulacao e testes no dado de saida 
'''

#from netCDF4 import Dataset
import pandas as pd


def getXYcoordinate(lat,lon,nc = None, XLAT=None, XLONG=None ):
    '''
    Retorna a coordenadas x,y de uma respectiva grade a partir da lat e long
    input:
        lat = latitude do ponto
        lon = longitude do ponto
        nc = arquivo wrfoutput em formato netCDF
        XLAT = latitude 2D
        XLONG = longitude 2D
    
    output:
        x , y  coordenadas
        
    exemplo:
        x, y = getXYcoordinate(-43.43,-22.32, nc = nc)
        x, y = getXYcoordinate(-43.43,-22.32, XLAT = XLAT, XLONG = XLONG)        
    '''
    try:
        if XLAT==None and XLONG==None:
            cor_lat = pd.DataFrame(nc.variables['XLAT'][0][:])
            cor_lat2 = pd.DataFrame({'a': cor_lat.iloc[:, 0], 'b': abs(cor_lat.iloc[:, 0] - lat)})
            x = cor_lat2[cor_lat2.b == min(cor_lat2.b)].index.to_numpy()[0]

            cor_lon = pd.DataFrame(nc.variables['XLONG'][0][:])
            cor_lon2 = pd.DataFrame({'a': cor_lon.iloc[0, :], 'b': abs(cor_lon.iloc[0, :] - lon)})
            y = cor_lon2[cor_lon2.b == min(cor_lon2.b)].index.to_numpy()[0]
            
        else:
            cor_lat = pd.DataFrame(XLAT)
            cor_lat2 = pd.DataFrame({'a': cor_lat.iloc[:, 0], 'b': abs(cor_lat.iloc[:, 0] - lat)})
            x = cor_lat2[cor_lat2.b == min(cor_lat2.b)].index.to_numpy()[0]

            cor_lon = pd.DataFrame(XLONG)
            cor_lon2 = pd.DataFrame({'a': cor_lon.iloc[0, :], 'b': abs(cor_lon.iloc[0, :] - lon)})
            y = cor_lon2[cor_lon2.b == min(cor_lon2.b)].index.to_numpy()[0]
            
    except:
        print( 'Insira um arquivo netcdf wrfouput ou os campos xlat e xlong em 2d numpy array')

    return  x , y

def extractpoint(x, y, array, convert = True):
    '''
    A partir das coordenadas de grade x y extrai serie historica do ponto de grade para uma variavel desejada numa numpy 3d array
    input:
        x ,y = coordenadas do ponto de grade
        array = numpy array 3d com indices t,x,y, onde t e o tempo        
        convert = Se True, converte o campo de temperatura de kelvin para graus celsios
    '''

    if convert==True:
        vlr = array[:, x, y]-273.15 #This change from kelvin to celsius

    else:
        vlr = array[:, x, y] #This change from kelvin to celsius

    return vlr


def extractpointNCDF(x, y, nc,  variavel = 'T2', convert = True):
    '''
    A partir das coordenadas de grade x y extrai serie historica do ponto de grade para uma variavel desejada num arquivo netcdf
    input:
        x ,y = coordenadas do ponto de grade
        nc = Arquivo netcdf desejado
        variavel = varaivel do netCDF desejada
        convert = Se True, converte o campo de temperatura de kelvin para graus celsios
    '''

    if convert==True:
        vlr = (nc.variables[variavel][:, x, y]-273.15)[:] #This change from kelvin to celsius

    else:
        vlr = (nc.variables[variavel][:, x, y])[:] #This change from kelvin to celsius

    return vlr


def extractvariavel(netCDFfile,variavel = 'T2'):
    nc = Dataset(netCDFfile)
    variavel = (nc.variables[variavel][:])[:]
    return variavel


def DataframePoints(est,simulacao, nc,start = '2018-01-17 00:00:00'):
    '''
    Retorna um dataframe com a variavel do campo simulacao para cada estacao meteorologica
    input:
        est: dataframe com os atributos Estacoes (nome da estacao),LAT e LON correspondente
        simulacao: array 3d da variavel de interesse
        nc: wrfouput em formato netcdf
        start: data inicial das simulacoes
        
    output:
        pandas dataframe: Columns = nome das estacoes, Linhas = data/hora correspondente
    '''
    estacoes = est.Estacoes
    lat = est.LAT.values
    lon = est.LON.values
    VAR = {}    

    for i , e in enumerate(estacoes):
        x , y = getXYcoordinate(lat[i],lon[i],nc)
        vlr = extractpoint(x,y,simulacao)        
        VAR[estacoes[i]] = vlr
    
    index = pd.date_range(start = start, freq = 'H',periods = len(simulacao))
    VAR = pd.DataFrame.from_dict(VAR)
    VAR.index = index
    
    return VAR
    
    
