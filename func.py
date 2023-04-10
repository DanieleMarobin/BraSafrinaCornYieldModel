import re
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import Weather as uw
import GLOBAL as GV

# Data
def get_CONAB_df():
    url = 'https://portaldeinformacoes.conab.gov.br/downloads/arquivos/SerieHistoricaGraos.txt'

    rename_conab_cols= {
        'produtividade_mil_ha_mil_t'    : 'Yield',
        'producao_mil_t'                : 'Production',
        'area_plantada_mil_ha'          : 'Area',
        'uf'                            : 'State',
        'produto'                       : 'Product',
        'id_produto'                    : 'Product_id',
        'ano_agricola'                  : 'CropYear',
        'dsc_safra_previsao'            : 'Crop',
        }

    url = url.replace(" ", "%20")
    df = pd.read_csv( url,low_memory=False,sep=';')

    df=df.rename(columns=rename_conab_cols)

    df['Crop']=df['Crop'].str.strip()
    df['Product']=df['Product'].str.strip()
    df['year']=(df['CropYear'].str[:4]).astype('int')+1 # Incresing by 1 to match the Modeling Nomenclature
    df = df.set_index('year', drop=False)
    df.index.name=''
    return df

def get_BRA_conab_data(states=['NATIONAL'], product='MILHO', crop='1ª SAFRA', years=list(range(1800,2050)), cols_subset=[], conab_df=None):
    if conab_df is None:
        df = get_CONAB_df()
    else:
        df = conab_df

    # Crop selection
    mask = (df['Product']==product) & (df['Crop']==crop)
    df=df[mask]

    # States selection
    if len(states)==0:
        df=df
    elif states[0]=='NATIONAL':        
        df=df.groupby(by='year').sum()
        df['Yield']=df['Production']/df['Area']
        df.index.name=''
        df['year']=df.index
    else:
        mask = np.isin(df['State'],states)
        df=df[mask]

    # Years selection
    mask = np.isin(df['year'],years)
    df=df[mask]

    # Column selection
    if len(cols_subset)>0: df = df[cols_subset]
    df=df.sort_values(by='year',ascending=True)
    
    return df

def get_BRA_prod_weights(states=[], product='MILHO', crop='1ª SAFRA', years=list(range(1800,2050)), conab_df=None):
    # rows:       years
    # columns:    region

    fo=get_BRA_conab_data(states=states, product=product, crop=crop, years=years, conab_df=conab_df)

    fo = pd.pivot_table(fo,values='Production',index='State',columns='year')
    fo.index=['BRA-'+s for s in fo.index]

    fo=fo/fo.sum()

    return fo.T

# Modeling

def prediction_interval(season_start, season_end, trend_case, full_analysis):
    if full_analysis:
        pred_date_start = season_start
        pred_date_end = season_end
    else:
        if trend_case:
            pred_date_start = season_start
            pred_date_end = season_start
        else:
            pred_date_start = season_end
            pred_date_end = season_end

    return pred_date_start, pred_date_end

def Build_DF_Instructions(WD_All='weighted', WD = GV.WD_HIST, prec_units = 'mm', temp_units='C', ext_mode = GV.EXT_DICT, ref_year = GV.CUR_YEAR, ref_year_start= dt(GV.CUR_YEAR,1,1)):
    fo={}

    if WD_All=='simple':
        fo['WD_All']='w_df_all'
    elif WD_All=='weighted':
        fo['WD_All']='w_w_df_all'

    fo['WD']=WD # which Dataset to use: 'hist', 'hist_gfs', 'hist_ecmwf', 'hist_gfsEn', 'hist_ecmwfEn'
        
    if prec_units=='mm':
        fo['prec_factor']=1.0
    elif prec_units=='in':
        fo['prec_factor']=1.0/25.4

    if temp_units=='C':
        fo['temp_factor']=1.0
    elif temp_units=='F':
        fo['temp_factor']=9.0/5.0

    fo['ext_mode']=ext_mode
    fo['ref_year']=ref_year
    fo['ref_year_start']=ref_year_start
    return fo

def var_windows_from_cols(cols=[], ref_year_start= dt(GV.CUR_YEAR,1,1)):
    """
    Typical Use:
        ww = um.var_windows_from_cols(m.params.index)
    
    Future development:
        - Use the other function 'def windows_from_cols(cols=[]):' to calculate the windows in this one
        - Note: 'def windows_from_cols(cols=[]):' just calculates the windows 
    """
    # Make sure that this sub is related to the function "def windows_from_cols(cols,year=2020):"
    var_windows=[]
    year = GV.LLY

    for c in (x for x  in cols if '-' in x):
        split=re.split('_|-',c)
        var = split[0]+'_'+split[1]
        
        if len(split)>1:
            d_start = dt.strptime(split[2]+str(year),'%b%d%Y')
            d_end = dt.strptime(split[3]+str(year),'%b%d%Y')

            start = uw.seas_day(d_start, ref_year_start)
            end = uw.seas_day(d_end, ref_year_start)
        
        var_windows.append({'variables':[var], 'windows':[{'start': start,'end':end}]})
    
    # I return 'np.array' to be able to use masks with it
    return np.array(var_windows)

def extract_yearly_ww_variables(w_df, var_windows=[], ref_year = GV.CUR_YEAR, ref_year_start= dt(GV.CUR_YEAR,1,1), join='inner', drop_na=True, drop_how='any'):
    w_df = uw.add_seas_year(w_df, ref_year, ref_year_start) # add the 'year' column
    w_df['seas_day'] = [uw.seas_day(d, ref_year_start) for d in w_df.index]

    wws=[]
    
    for v_w in var_windows:    
        # Get only needed variables and 'year','seas_day'
        #    1) 'seas_day': to select the weather window
        #    2) 'year': to be able to group by crop year

        w_cols=['year','seas_day']
        w_cols.extend(v_w['variables'])
        w_df_sub = w_df[w_cols]
        
        for w in v_w['windows']:
            s = w['start']
            e = w['end']
            id_s = uw.seas_day(date=s, ref_year_start=ref_year_start)
            id_e = uw.seas_day(date=e, ref_year_start=ref_year_start)

            ww = w_df_sub[(w_df_sub['seas_day']>=id_s) & (w_df_sub['seas_day']<=id_e)]
            ww=ww.drop(columns=['seas_day'])
            ww.columns=list(map(lambda x:'year'if x=='year'else x+'_'+s.strftime("%b%d")+'-'+e.strftime("%b%d"),list(ww.columns)))
            ww = ww.groupby('year').mean()
            ww.index=ww.index.astype(int)
            wws.append(ww)  
                           
    w_df=w_df.drop(columns=['year','seas_day'])
    out_df = pd.concat(wws, sort=True, axis=1, join=join)        
    if drop_na: out_df.dropna(inplace=True, how=drop_how) # how : {'any', 'all'}
    return  out_df

# Charting

def add_series(fig,x,y,name=None,mode='lines+markers',showlegend=True,line_width=1.0,color='black',marker_size=5,legendrank=0):
    fig.add_trace(go.Scatter(x=x, y=y,mode=mode, line=dict(width=line_width,color=color), marker=dict(size=marker_size), name=name, showlegend=showlegend, legendrank=legendrank))

def line_chart(x,y,name=None,mode='lines+markers',showlegend=True,line_width=1.0,color='black',marker_size=5,legendrank=0,width=1400,height=600):
    fig = go.Figure()
    add_series(fig,x,y,name,mode=mode,showlegend=showlegend,line_width=line_width,color=color,marker_size=marker_size,legendrank=legendrank)
    update_layout(fig,marker_size,line_width,width,height)
    return fig

def update_layout(fig,marker_size,line_width,width,height):
    fig.update_traces(marker=dict(size=marker_size),line=dict(width=line_width))
    fig.update_xaxes(tickformat="%d %b")
    fig.update_layout(autosize=True,font=dict(size=12),hovermode="x unified",margin=dict(l=20, r=20, t=50, b=20))
    fig.update_layout(width=width,height=height)

def seas_day(date, ref_year_start= dt(GV.CUR_YEAR,1,1)):
    """
    'seas_day' is the X-axis of the seasonal plot:
            - it makes sure to include 29 Feb
            - it is very useful in creating weather windows
    """

    start_idx = 100 * ref_year_start.month + ref_year_start.day
    date_idx = 100 * date.month + date.day

    if (start_idx<300):
        if (date_idx>=start_idx):
            return dt(GV.LLY, date.month, date.day)
        else:
            return dt(GV.LLY+1, date.month, date.day)
    else:
        if (date_idx>=start_idx):
            return dt(GV.LLY-1, date.month, date.day)
        else:
            return dt(GV.LLY, date.month, date.day)

def chart_actual_vs_model(model, df, y_col, x_col=None, plot_last_actual=False, height=None):
    '''
    plot_last_actual=False
        - sometimes the last row is the the prediction (so it is better not to show it as 'actual')
    '''
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1,0.4])

    if x_col is None:
        x=df.index
    else:
        x=df[x_col]

    y_actu=df[y_col]
    y_pred= model.predict(df[model.params.index])
    y_diff=100*(y_pred-y_actu)/y_actu    

    if plot_last_actual:
        x_actu=x[:]
    else:
        mask= y_actu.index <(y_actu.index.max())
        x_actu=x[mask]
        y_actu=y_actu[mask]    
        
    fig.add_trace(go.Scatter(x=x_actu, y=y_actu,mode='lines+markers', line=dict(width=1,color='black'), marker=dict(size=5), name='Actual'), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=y_pred,mode='lines+markers', line=dict(width=1,color='blue'), marker=dict(size=5), name='Model'), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=y_diff, name='Error (%)'), row=2, col=1)

    hovermode='x unified' # ['x', 'y', 'closest', 'x unified', 'y unified']

    fig.update_layout(height=height, hovermode=hovermode)

    return fig
        
def waterfall(yield_contribution):
    df= yield_contribution.loc['Difference':'Difference']
    df=df.T

    # Remove the 0s (because they give no contrinution to the yield change)
    mask = abs(df['Difference'])>0
    df=df[mask]
    df=df.T

    sorted_cols=[c for c in df.columns if c!='Yield']
    sorted_cols.append('Yield')
    df=df[sorted_cols]

    measure=['relative']*(len(sorted_cols)-1)
    measure.append('total')

    y=list(df.values[0])
    text = ['+'+ str(round(v,3)) if v > 0 else '-'+ str(abs(round(v,3))) for v in y]
    text[-1]= 'Yield Difference vs Trend: <br>'+text[-1]

    if y[-1]<0:
        totals = {"marker":{"color":"darkred", "line":{"color":"red", "width":3}}}
    else:
        totals = {"marker":{"color":"darkgreen", "line":{"color":"green", "width":3}}}


    fig = go.Figure(go.Waterfall(
        orientation = 'v',
        measure = measure,
        x = sorted_cols,
        textposition = 'auto',
        text = text,
        y = y,
        totals=totals,
        # connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
                
    return fig

def visualize_model_ww(model, ref_year_start, train_df=None, fuse_windows=True):
    '''
    fuse_windows = True:
        - it creates the aggregate of all the different windows of a certain variable:
        - if there are 3 variables:
            1) Precipitation in Jan-Feb
            2) Precipitation in Feb-Mar
            3) Precipitation in Mar-Apre
                -> it will create a single line called Precipitation that will sum all the coefficients in the overlapping parts
    '''

    if train_df is None:
        data=[1]*len(model.params)
        train_df_mean= pd.Series(data=data, index=model.params.index)
    else:
        train_df_mean=train_df.mean()
    fig = go.Figure()
    year = GV.LLY
    var_dict={}
    legend=[]

    for c in (x for x  in model.params.index if '-' in x):
        split=re.split('_|-',c)
        v = split[0]+'_'+split[1]
        coeff = model.params[c]           

        if len(split)>1:
            d_start = dt.strptime(split[2]+str(year),'%b%d%Y')
            d_end = dt.strptime(split[3]+str(year),'%b%d%Y')

            start = seas_day(d_start, ref_year_start)
            end = seas_day(d_end, ref_year_start)

            index = (np.arange(start, end + timedelta(days = 1), dtype='datetime64[D]'))
            data = np.full(len(index), coeff*train_df_mean[c])
            
            if v in var_dict:
                var_dict[v].append(pd.Series(data=data,index=index))
            else:
                var_dict[v]=[pd.Series(data=data,index=index)]
                
    for v, series_list in var_dict.items():
        if ('Temp' in v):
            color='orange'
        elif ('Sdd' in v):
            color='red'                        
        else:
            color='blue'

        name_str = '   <b>'+str(v)+'</b>'
        y_str = '   %{y:.2f}'
        x_str = '   %{x|%b %d}'
        hovertemplate="<br>".join([name_str, y_str, x_str, "<extra></extra>"])
    
        if fuse_windows:
            var_coeff=pd.concat(series_list,axis=1).sum(axis=1)
            var_coeff=var_coeff.resample('1D').asfreq()
            fig.add_trace(go.Scatter(x=var_coeff.index , y=var_coeff.values, name=v,mode='lines', line=dict(width=2,color=color, dash=None), marker=dict(size=8), showlegend=True, hovertemplate=hovertemplate))
        else:
            for sl in series_list:
                if v in legend:
                    showlegend=False
                else:
                    showlegend=True
                legend.append(v)

                if sl.values[0]<0.0:
                    dash='dash'
                else:
                    dash=None

                x=[sl.index[0],sl.index[-1]]
                y=[sl.values[0],sl.values[-1]]
                fig.add_trace(go.Scatter(x=x, y=y, name=v,mode='lines+markers', line=dict(width=2,color=color, dash=dash), marker=dict(size=8), showlegend=showlegend, hovertemplate=hovertemplate))

    # add today line
    fig.add_vline(x=seas_day(dt.today(), ref_year_start).timestamp() * 1000, line_dash="dash",line_width=1, annotation_text="Today", annotation_position="bottom")
    
    hovermode='x unified' # ['x', 'y', 'closest', 'x unified', 'y unified']
    fig.update_layout(height=750, legend=dict(orientation="h",yanchor="bottom",y=1.1,xanchor="left",x=0), hovermode=hovermode)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_xaxes(tickformat="%d %b")
    return fig
