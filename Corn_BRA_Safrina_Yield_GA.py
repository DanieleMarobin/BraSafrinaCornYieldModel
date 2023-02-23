"""
This file relies on the library 'pygad' for the Genetic Algorithms calculations
Unfortunately there are certain functions that do not accept external inputs
so the only way to pass variables to them is to have some global variables

### Conab:

Corn (Milho): https://www.conab.gov.br/info-agro/safras/serie-historica-das-safras/itemlist/category/910-Milho \
Download Page: https://portaldeinformacoes.conab.gov.br/download-arquivos.html (look for "Serie Histórica Grãos") \
Actual file: https://portaldeinformacoes.conab.gov.br/downloads/arquivos/SerieHistoricaGraos.txt \
Human files: https://www.conab.gov.br/info-agro and then select SAFRAS \
Probably ending up here: https://www.conab.gov.br/info-agro/safras \

### IBGE:
for yearly municipality all corn: https://sidra.ibge.gov.br/Tabela/1612 \
for yearly municipality 1st and 2nd split: https://sidra.ibge.gov.br/Tabela/839 \

### Agriquest:

You can access your account here :
https://agriquest.geosys-na.com/Common/login

Login : AQ_AVERE_mln \
Password : AQ_AVERE_mln

### To install GDAL:
https://www.youtube.com/watch?v=8iCWUp7WaTk \
wheel from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal \
I downloaded the file: GDAL‑3.4.2‑cp39‑cp39‑win_amd64.whl because I have python 3.9.7 (and can be check by typing) \
python --verions in the cmd prompt

### To install Geopandas:
https://gis.stackexchange.com/questions/330840/error-installing-geopandas \
first need to install Fiona from here:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona \
choosing the file Fiona-1.8.21-cp39-cp39-win_amd64.whl because I have python 3.9.7 \
https://automating-gis-processes.github.io/CSC18/lessons/L2/geopandas-basics.html


### Genetic Algorithm:

https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html
"""

import sys

import re
sys.path.append(r'\\ac-geneva-24\E\grains trading\Streamlit\Monitor\\')
sys.path.append(r'C:\Monitor\\')

import os
from datetime import datetime as dt
from copy import deepcopy
import concurrent.futures

import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np

import statsmodels.api as sm


import Weather as uw
import func as fu

import GLOBAL as GV

import warnings; warnings.filterwarnings("ignore")

# General Settings
if True:
    ref_year=2023
    ref_year_start=dt(ref_year-1,9,1)

def Define_Scope():
    """
    'geo_df':
        it is a dataframe (selection of rows of the weather selection file)
    'geo_input_file': 
        it needs to match the way files were named by the API
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec.csv
            GV.WS_STATE_ALPHA   ->  MT_Prec.csv
            GV.WS_STATE_CODE    ->  51_Prec.csv

    'geo_output_column':
        this is how the columns will be renamed after reading the above files (important when matching weight matrices, etc)
            GV.WS_STATE_NAME    ->  Mato Grosso_Prec
            GV.WS_STATE_ALPHA   ->  MT_Prec
            GV.WS_STATE_CODE    ->  51_Prec
    """

    fo={}

    # Geography (Read the comment above, Expand the section if it is hidden/collapsed)
    w_sel_df = uw.get_w_sel_df()
    fo['geo_df'] = w_sel_df[w_sel_df[GV.WS_COUNTRY_ALPHA] == 'BRA']
    fo['geo_input_file'] = GV.WS_UNIT_ALPHA 
    fo['geo_output_column'] = GV.WS_UNIT_ALPHA

    # Weather Variables
    fo['w_vars'] = [GV.WV_PREC, GV.WV_TEMP_MAX, GV.WV_SDD_30]

    # Time
    fo['years']=list(range(1990,GV.CUR_YEAR+1))
    
    return fo

def Get_Data_Single(scope: dict, var: str = 'yield', fo = {}):    
    if (var=='yield'):
        df = fu.get_BRA_conab_data(states=['NATIONAL'], product='MILHO', crop='2ª SAFRA', years=scope['years'], cols_subset=['year','Yield'], conab_df=None)
        df = df.set_index('year', drop=False)
        return df

    elif (var=='weights'):        
        return fu.get_BRA_prod_weights(states=list(fo['locations']), product='MILHO', crop='2ª SAFRA', years=scope['years'], conab_df=None)

    elif (var=='w_df_all'):
        return uw.build_w_df_all(scope['geo_df'], scope['w_vars'], scope['geo_input_file'], scope['geo_output_column'])

    elif (var=='w_w_df_all'):
        return uw.weighted_w_df_all(fo['w_df_all'], fo['weights'], output_column='BRA', ref_year=ref_year, ref_year_start=ref_year_start)

    return fo

def Get_Data_All_Parallel(scope):
    # https://towardsdatascience.com/multi-tasking-in-python-speed-up-your-program-10x-by-executing-things-simultaneously-4b4fc7ee71e

    fo={}

    # Time
    fo['years']=scope['years']

    # Space
    fo['locations']=scope['geo_df'][GV.WS_STATE_ALPHA]
    # fo['locations']=['MT']

    download_list=['yield', 'weights', 'w_df_all']
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results={}
        for variable in download_list:
            results[variable] = executor.submit(Get_Data_Single, scope, variable, fo)
    
    for var, res in results.items():
        fo[var]=res.result()
    
    # Weighted Weather: it is here because it needs to wait for the 2 main in ingredients (1) fo['w_df_all'], (2) fo['weights'] to be calculated first
    variable = 'w_w_df_all'
    fo[variable] = Get_Data_Single(scope, variable, fo)

    return fo



def Build_DF(raw_data, instructions, saved_m):
    """
    The model DataFrame Columns:
            1) Yield (y)
            2) N Variables
            3) Constant (added to be able to fit the model with 'statsmodels.api')

            1+N+1 = 2+N Columns
    """

    w_all=instructions['WD_All'] # 'simple'->'w_df_all', 'weighted'->'w_w_df_all'
    WD=instructions['WD'] # weather Dataset: 'hist', 'hist_gfs', 'hist_ecmwf', 'hist_gfsEn', 'hist_ecmwfEn'
    ref_year=instructions['ref_year']
    ref_year_start=instructions['ref_year_start']

    w_df = raw_data[w_all][WD]
    wws = fu.var_windows_from_cols(saved_m.params.index)
    model_df = fu.extract_yearly_ww_variables(w_df = w_df,var_windows= wws, ref_year=ref_year, ref_year_start=ref_year_start)
    model_df = pd.concat([raw_data['yield'], model_df], sort=True, axis=1, join='inner')
    model_df = sm.add_constant(model_df, has_constant='add')

    return model_df
    
def Build_Pred_DF(raw_data, instructions,  date_start=dt.today(), date_end=None, trend_yield_case= False, saved_m=None):
    """

    """    
    
    dfs = []
    w_all=instructions['WD_All'] # 'simple'->'w_df_all', 'weighted'->'w_w_df_all'
    WD=instructions['WD'] # weather Dataset to extend: 'hist', 'hist_gfs', 'hist_ecmwf', 'hist_gfsEn', 'hist_ecmwfEn'
    ref_year=instructions['ref_year']
    ref_year_start=instructions['ref_year_start']
    ext_mode=instructions['ext_mode'] # Extention modes ('temperature' extend with 'mean', 'precipitation' extend with 'mean')

    w_df = raw_data[w_all][WD]
    raw_data_pred = deepcopy(raw_data)

    # when calling 'Build_DF' --> model_df = pd.concat([raw_data['yield'], model_df], sort=True, axis=1, join='inner')
    # as it is "join='inner'" if the 'ref_year' (prediction year) is not in the dataframe then all the data for the ref_year is deleted
    if ref_year not in raw_data_pred['yield'].index:
        raw_data_pred['yield'].loc[ref_year,'year']=ref_year

    if (date_end==None): date_end = w_df.index[-1] # this one to check well what to do
    days_pred = list(pd.date_range(date_start, date_end))

    for i, day in enumerate(days_pred):
        if trend_yield_case:
            keep_duplicates='last'
        else:
            keep_duplicates='first'

        # Extending the Weather        
        if (i==0):
            # Picks the extension column and then just uses it till the end            
            raw_data_pred[w_all][WD], dict_col_seas = uw.extend_with_seasonal_df(w_df[w_df.index<=day], return_dict_col_seas=True, var_mode_dict=ext_mode, ref_year=ref_year, ref_year_start=ref_year_start, keep_duplicates=keep_duplicates)
        else:
            raw_data_pred[w_all][WD] = uw.extend_with_seasonal_df(w_df[w_df.index<=day], input_dict_col_seas = dict_col_seas, var_mode_dict=ext_mode, ref_year=ref_year, ref_year_start=ref_year_start, keep_duplicates=keep_duplicates)
        
        # Build the 'Simulation' DF    
        w_df_pred = Build_DF(raw_data_pred, instructions, saved_m) # Take only the GV.CUR_YEAR row and append

        # Append row to the final matrix (to pass all at once for the daily predictions)
        dfs.append(w_df_pred.loc[ref_year:ref_year])
    
    fo = pd.concat(dfs)

    # This one is to be able to have the have the 'full_analysis' chart
    # and also it makes a lot of sense:
    #       - the output 'fo' shows for each row (day), which dataset should be used
    fo.index= days_pred.copy()
    return fo