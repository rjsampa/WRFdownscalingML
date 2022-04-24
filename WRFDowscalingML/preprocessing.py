from netCDF4 import Dataset 
import numpy as np
import pandas as pd


#------------------------------------------------------------------------
"""
Isola variaveis dependentes e independentes do arquivo wrfoutput

"""
import matplotlib.tri as tri
import matplotlib.pyplot as plt
#import geopandas as geo
#lulc= pd.read_json('../Dados/interin/X_d3.json')
#rj = geo.read_file('../../../dados_informacoes/vetores/RMRJ/RMRJ.shp')


def plot_map(campo,figsize=(10,5),
             cmap='jet',title = None, 
             lat=None,lon=None, 
             linecolor='lightgrey',
             vetor = None,
             ax=None,
             mask_water=False,
             lulc=None
            ):

    campo = campo.flatten()
    modelo = np.nan_to_num(campo,nan=0)
    
    triang = tri.Triangulation(lon,lat)
    if mask_water:
        mask = (lulc==17).flatten()
        mask = np.all(np.where(mask[triang.triangles], True, False), axis=1)
        triang.set_mask(mask)
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    
    cax = ax.tricontourf(triang,modelo,cmap=cmap)
    ax.set_xlabel('Long',fontsize=14)
    ax.set_ylabel('Lat',fontsize=14)
    #ax.set_ylim(-23.2,-22.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    #fig.colorbar(cax,label='Intensidade das ICUs',ax=ax)
    plt.colorbar(cax, ax=ax, shrink=0.7)
    
    #rj.plot(facecolor='none', edgecolor= linecolor, lw =0.7,ax=ax)
    
    if title is not None:
        ax.set_title(title)  



def rotT2(T2):
    """Rotaciona a imagem T2 do wrf para um o padrao Norte superior"""
    Rep = np.rot90(T2,2)
    Rep = np.flip(Rep,1)
    return Rep

class DataBase:
    def __init__(self,wrfoutput):
        """
        Abre o arquivo netcdf wrfout(..) e extrai as variaveis de interese
        input: 
            wrfouput: nome ou diretorio para o arquvio wrfout(..) entre aspas
        """
        self.wrfoutput = Dataset(wrfoutput)
        try:
            self.lulc = self.get_var_static('LU_INDEX').flatten()
            self.lat = self.get_var_static('XLAT').flatten()
            self.lon = self.get_var_static('XLONG').flatten()
            
        except:
            print('Lat and Lon info not loading!')
        
    @property
    def list_variables(self):
        return self.wrfoutput.variables.keys()
      
    def Temperatura(self):
        """
        Retorna a temperatura a 2m referente ao aquivo wrfouput 
        """        
        self.T2m = self.wrfoutput.variables['T2'][:]
        t2rot = []
        for i , n in enumerate(self.T2m):
            campo = rotT2(n)
            t2rot.append(campo)
        self.T2m = np.array(t2rot)
        return self.T2m
    
    def get_var_din(self, name, steps=None):
        """
        get dynamic variable in database 
        
        name: str , default obligatory
            variable name
        
        range:  tuple , default None
            temporal range of the variable select
        """        
        if steps is None:
            self.var = self.wrfoutput.variables[name][:]
        else:
            self.var = self.wrfoutput.variables[name][steps[0]:steps[1]]
        
        varRot = []
        for i , n in enumerate(self.var):
            campo = rotT2(n)
            varRot.append(campo)
        varGet = np.array(varRot)
        return varGet
    
    def get_var_static(self,name):
        """
        get static variable 
        """
        self.static = self.wrfoutput.variables[name][1]
        return rotT2(self.static)
    

    def get_table_independent_var(self, variables:list = None):
        df = {}#pd.DataFrame()
        for i in variables:
            var = np.array(self.get_var_static(i))#.flatten()
            df[i] = var
        return df
    
    def get_times_periods(self):
        d0 = pd.to_datetime(self.wrfoutput.SIMULATION_START_DATE,format='%Y-%m-%d_%H:%M:%S')
        serie = pd.date_range(start=d0,freq='H',periods=self.wrfoutput.variables['Times'].shape[0])
        return serie
        

    def plot(self,campo, water_mask=True, **kw):
        plot_map(campo,lat=self.lat,lon=self.lon, mask_water=water_mask, lulc=self.lulc, **kw)
    

    
####  variaveis do terreno
def slope_aspect(mde):    
    '''
    Retorno a declividade e aspecto do terreno
    input:
        mde: array do modelo digital de elevacao 
        
    output:
        slope: 2D array 
        aspect: 2D array  
    '''
    x, y = np.gradient(mde)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    return np.array(slope) , aspect

def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2


# ------------------ Criando dataframe Test e treino -------------------
#funcoes
def flatten(array):
    '''
    traforma uma array nD numa array 1D 
    input: 
        array: numpy array nDimensional
    output:
        array numpy 1D'''
    
    return array.flatten()
    

def time_index(start='2017-12-17  00:00:00', periods=67):    
    """ 
    Define o Dataindex para o passo X, considerando o intervalo do estudo
    necessario ajustar o inicio para i periodo desejado
    input:
        start: data e hora de inicio
        periods = numero de periodos desejados 
        
    output:
        time serie
    """
    time = pd.date_range(start = start,
                  periods=periods,freq='H')
    return time

def pandasDF(lista, nomes,start='2017-12-17  00:00:00'):
    '''
    Cria um pandas daframe a partir de uma lista com as variaveis de interesse
    input:
        lista: lista/tupla com variaveis de interesse em 2d
        nomes: lista com nomes das variaveis de interesse
    
    output:
        Pandas Daframe com variaveis de interesse     
    '''
    df = pd.DataFrame()
    for i , var in enumerate(lista):
        df[nomes[i]] = var
        
    
    return df
        
        