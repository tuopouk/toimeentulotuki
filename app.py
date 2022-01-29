#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from dash import dcc
from dash import html
import dash_daq
from flask import Flask
import os
import base64
import io
from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State
from dash.exceptions import PreventUpdate
import random
import dash_bootstrap_components as dbc
import time
from datetime import datetime
import io
import holidays
from tqdm import tqdm



spinners = ['graph', 'cube', 'circle', 'dot' ,'default']

features = ['edellinen', 'pv_nro', 'kuukausi_nro']


external_stylesheets = [
                        "https://bootswatch.com/5/morph/bootstrap.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
                        'https://codepen.io/chriddyp/pen/brPBPO.css'
                       ]


server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, 
           prevent_initial_callbacks = False, 
           server = server,
           meta_tags = [{'name':'viewport',
                        'content':'width=device-width, initial_scale=1.0, maximum_scale=1.2, minimum_scale=0.5'}],
           external_stylesheets = external_stylesheets
          )


app.title = 'Toimeentulotuki Suomessa'
app.scripts.append_script({"external_url": "https://cdn.plot.ly/plotly-locale-fi-latest.js"})

# app.index_string = '''
# <!DOCTYPE html>
# <html>
#     <head>
#         {%metas%}
#         <title>{%title%}</title>
#         {%favicon%}
#         {%css%}
#     </head>
#     <body>
        
#         {%app_entry%}
#         <footer>
#             {%config%}
#             {%scripts%}
#             <script src="https://cdn.plot.ly/plotly-locale-fi-latest.js"></script>
# <script>Plotly.setPlotConfig({locale: 'fi'});</script>
#           {%renderer%}
#           </footer>
          
#             </body>
# </html>
'''

# Haetaan Kelan toimeentulotukidata.
def get_kela_data():
    
    kela = pd.read_csv('https://www.avoindata.fi/data/dataset/4b64be55-5a69-4f6b-a9d5-d4cbbe5c4382/resource/6409bdec-4a48-46a9-8729-5a727e37cd55/download/data.csv')

    
    kela.maksu_pv = pd.to_datetime(kela.maksu_pv)
    kela = kela.set_index('maksu_pv')
    kela = kela.sort_index()
    kela.drop(['vuosikuukausi','vuosi','kunta_nro','etuus'],axis=1,inplace=True)
    
    koko_maa = kela.copy()
    koko_maa = koko_maa.reset_index().groupby(['maksu_pv', 'kuukausi_nro', 'palautus']).sum()
    koko_maa = koko_maa.reset_index().set_index('maksu_pv')
    koko_maa['kunta_nimi'] = 'Koko maa'
    
    kela = pd.concat([kela,koko_maa])
    kela = kela.sort_index()
    
    return kela



# Muokataan dataa niin, että maksut ja palautukset ovat omina sarakkeinaan.
def get_combined_data(kela):
    
    palautukset = kela[kela.palautus=='Kyllä'].copy()
    maksut = kela[kela.palautus=='Ei'].copy()
    palautukset.drop(['palautus','kuukausi_nro'],axis=1, inplace=True)
    maksut.drop('palautus',axis=1, inplace=True)
    palautukset.maksettu_eur = -1*palautukset.maksettu_eur
    palautukset.valtio_eur = -1*palautukset.valtio_eur
    palautukset.valtio_kunta_eur = -1*palautukset.valtio_kunta_eur

    df = pd.merge(left = maksut.reset_index(), right = palautukset.reset_index().rename(columns = {'maksettu_eur':'palautus_eur',
                                                                                             'valtio_eur':'palautus_valtio_eur',
                                                                                             'valtio_kunta_eur':'palautus_valtio_kunta_eur',
                                                                                                                                                                                                        }), 
             on = ['maksu_pv','kunta_nimi'], how = 'outer').sort_values(by=['maksu_pv','kunta_nimi']).set_index('maksu_pv').fillna(0)
    
    df['netto_eur'] = df['maksettu_eur'] - df['palautus_eur']
    df['valtio_netto_eur'] = df['valtio_eur'] - df['palautus_valtio_eur']
    df['valtio_kunta_netto_eur'] = df['valtio_kunta_eur'] - df['palautus_valtio_kunta_eur']
    df['suoritukset_eur'] = df['maksettu_eur'] + df['palautus_eur']
    df['suoritukset_valtio_kunta_eur'] = df['palautus_valtio_kunta_eur'] + df['valtio_kunta_eur']
    df['suoritukset_valtio_eur'] = df['palautus_valtio_eur'] + df['valtio_eur']
    df = df.sort_index()
    
        
    return df



# Hae kunnan data kumulatiivisena.
def get_cum_city_data(kunta):
    
    df = data.copy()
    
    df = df[df.kunta_nimi == kunta].copy()
    
    df = df.sort_index()
    
    df['palautus_eur_kum'] = df.palautus_eur.cumsum()
    df['maksettu_eur_kum'] = df.maksettu_eur.cumsum()
    df['valtio_eur_kum'] = df.valtio_eur.cumsum()
    df['valtio_kunta_eur_kum'] = df.valtio_kunta_eur.cumsum()
    df['palautus_valtio_eur_kum'] = df.palautus_valtio_eur.cumsum()
    df['palautus_valtio_kunta_eur_kum'] = df.palautus_valtio_kunta_eur.cumsum()
    df['netto_eur_kum'] = df['netto_eur'].cumsum()
    
    df['valtio_netto_eur_kum'] = df['valtio_netto_eur'].cumsum()
    df['valtio_kunta_netto_eur_kum'] = df['valtio_kunta_netto_eur'].cumsum()
    df['suoritukset_eur_kum'] = df['suoritukset_eur'].cumsum()
    df['suoritukset_valtio_kunta_eur_kum'] = df['suoritukset_valtio_kunta_eur'].cumsum()
    df['suoritukset_valtio_eur_kum'] = df['suoritukset_valtio_eur'].cumsum()
    df['kunta'] = kunta
    df['pv_nro'] = [int(c[-1]) for c in df.index.astype(str).str.split('-')]

    return df


# Apumetodi, jolla siirretään päiviä eteenpäin.
def shift_dates(dataset, label):
    
    ds = dataset.copy()
    ds['edellinen'] = ds[label].shift(periods=1)
    f = features.copy()
    f.append(label)
    ds = ds[f].dropna()
    return ds

# Onko päivä julkinen pyhäpäivä Suomessa?
def is_public_holiday(date):
    
    date = pd.to_datetime(date)

    return date in holidays.FIN(years=date.year)

# Onko päivä viikonloppuna?
def is_weekend(date):
    
    date = pd.to_datetime(date)
    
    return date.weekday() >= 5


# Onko päivä arkipäivä?
def is_bdate(date):
    
    date = pd.to_datetime(date)
    
    return not (is_weekend(date) or is_public_holiday(date))

# Palautetaan seuraava arkipäivä.
def next_wanted_weekday(date, threshold):
    
    date = pd.to_datetime(date)
    
    wanted_day = date + pd.Timedelta(days = threshold)
    
    if is_bdate(wanted_day):
        return wanted_day
    
    else:
        while not is_bdate(wanted_day):
            wanted_day = wanted_day + pd.Timedelta(days = 1)
    return wanted_day

# Muutetaan päivämäärä kvartaaleiksi.
def to_quartals(date):
    year = str(date).split('-')[0]
    month = str(date).split('-')[1]
    return year +' '+{'03':'Q1','06':'Q2','09':'Q3','12':'Q4'}[month]


# Jaetaan data opetus, validointi ja testidataan.
# Optimoidaan hyperparametri validointidatalla
# ja testataan lopullinen algoritmi testidatalla.
# Palautetaan tulosmatriisi.
def train_val_test(dataset, label, test_size=.3):
    
    # Train - test -jako
    
    train_size = 1 - test_size
    train_data = dataset.iloc[:int(train_size*len(dataset)),:]
    test_data = dataset.iloc[int(train_size*len(dataset)):,:]
    val_data = test_data.iloc[:int(len(test_data)/2),:]
    test_data = test_data.iloc[int(len(test_data)/2):,:]
    
    df = train_data.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan
    next_day = val_data.index.values[0]
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')
    
    
    
    alpha_list = []
    
    # Alpha saa arvoja väliltä [2**-15, 2**15].
    # Alphaa sanotaan useimmin lambdaksi. Scikit-learnissa se on kuitenkin alpha.
    
    for alpha in tqdm([2**c for c in range(-15,16)]):

            
            model = Ridge(random_state=42, alpha = alpha)
            scl = StandardScaler()

            x_train = train_data[features]
            X_train = scl.fit_transform(x_train)
            y_train = train_data[label]

            model.fit(X_train,y_train)
            df[label] = np.maximum(df.edellinen,model.predict(scl.transform(df[features])))


            dfs = []
            dfs.append(df)

            days = len(val_data)
                       

            for i in range(1,days):

                dff = dfs[-1].copy()

                dff.edellinen = dff[label]
                dff[label] = np.nan
                next_day = val_data.index.values[i]
                dff['maksu_pv'] = next_day
                dff = dff.set_index('maksu_pv')
                dff['kuukausi_nro'] =dff.index.month
                dff['pv_nro'] =dff.index.day

                dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features])))
                dfs.append(dff)
            error = np.absolute(val_data.iloc[-1][label]-pd.concat(dfs).iloc[-1][label])
            alpha_list.append({'alpha':alpha, 'error':error})
      
    alpha = pd.DataFrame(alpha_list).sort_values(by='error').head(1).alpha.values[0]
    
    train_data_prev = train_data.copy()
    
    train_data = pd.concat([train_data,val_data])
                    
    model = Ridge(random_state=42, alpha = alpha)
    scl = StandardScaler()

    x_train = train_data[features]
    X_train = scl.fit_transform(x_train)
    y_train = train_data[label]

    model.fit(X_train,y_train)
        
        
    df = train_data.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan    
    next_day = test_data.index.values[0]
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')                 

    df[label] = np.maximum(df.edellinen,model.predict(scl.transform(df[features])))

    dfs = []
    dfs.append(df)

    days = len(test_data)

    for i in range(1,days):

        dff = dfs[-1].copy()

        dff.edellinen = dff[label]
        dff[label] = np.nan
        next_day = test_data.index.values[i]
        
        
        dff['maksu_pv'] = next_day
        dff = dff.set_index('maksu_pv')
        dff['kuukausi_nro'] =dff.index.month
        dff['pv_nro'] =dff.index.day

        dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features])))
        dfs.append(dff)
        
    test_data['ennustettu'] = pd.concat(dfs)[label]
    
    train_data_prev['split'] = 'train'
    val_data['split'] = 'val'
    test_data['split'] = 'test'
    
  
    
    
    result = pd.concat([train_data_prev, val_data, test_data])
    result['alpha'] = alpha
    result['split_portion'] = test_size
    
    return result

# Tuotetaan ennuste halutulle ajalle valitulla alpha-parametrilla.
def predict(dataset, label, length, alpha):
    
    
    model = Ridge(random_state = 42, alpha = alpha)
    scl = StandardScaler()
    x = dataset[features]
    y = dataset[label]
    
    X = scl.fit_transform(x)
    
    model.fit(X,y)
    
    df = dataset.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan
    next_day = next_wanted_weekday(pd.to_datetime(df.index.values[0]), threshold=1)
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')
    
    df[label] = np.maximum(df.edellinen,model.predict(scl.transform(df[features])))
    
    
    dfs = []
    dfs.append(df)
    
    current_date = df.index.values[0]
    last_date = current_date + pd.Timedelta(days = length)
        
    while current_date < last_date:

        dff = dfs[-1].copy()

        dff.edellinen = dff[label]
        dff[label] = np.nan
        next_day = next_wanted_weekday(pd.to_datetime(current_date), threshold=1)
        dff['maksu_pv'] = next_day
        dff = dff.set_index('maksu_pv')
        dff['kuukausi_nro'] =dff.index.month
        dff['pv_nro'] =dff.index.day
        
        dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features])))
        dfs.append(dff)
        current_date = next_day
    prediction = pd.concat(dfs)                   
    
    dataset['forecast'] = 'Toteutunut'
    prediction ['forecast'] ='Ennuste'
    
    result_df = pd.concat([dataset,prediction])
    
    result_df['ennusteen_pituus'] = length
    
    return result_df


# Alustetaan lähtödata ja valikot.
keladata = get_kela_data()
data = get_combined_data(keladata)
kunta_options = [{'label':k, 'value':k} for k in sorted(pd.unique(data.kunta_nimi))]
labels = {'Maksut yhteensä':'maksettu_eur_kum',
             'Palautukset yhteensä':'palautus_eur_kum',
             'Valtion kokonaan suorittamat maksut': 'valtio_eur_kum',
             'Valtion kokonaan saamat palautukset': 'palautus_valtio_eur_kum',
             'Kuntien ja valtion puoliksi rahoittamat maksut': 'valtio_kunta_eur_kum',
             'Kuntien ja valtion puoliksi saadut palautukset': 'palautus_valtio_kunta_eur_kum',
             'Suoritukset yhteensä': 'suoritukset_eur_kum',
             'Nettomaksut yhteensä (maksut - palautukset)': 'netto_eur_kum',
             'Valtion nettomaksut': 'valtio_netto_eur_kum',
             'Kuntien ja valtion puoliksi rahoittamat nettomaksut':'valtio_kunta_netto_eur_kum',
             'Valtion kokonaan tehdyt suoritukset yhteensä': 'suoritukset_valtio_eur_kum',
             'Valtion ja kunnat puoliksi tehdyt suoritukset yhteensä':'suoritukset_valtio_kunta_eur_kum'}

label_options = [{'label':k, 'value':k} for k in labels.keys()]


# Visualisoidaan haluttu muuttuja päivittäin.
def plot_daily_data(kunta, label):
    
    df = data[data.kunta_nimi==kunta]

    l = labels[label].replace('_kum','')
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(df.iloc[i][l],2)).replace(',',' ')) for i in range(len(df))]

    
    figure = go.Figure(data = [
                            go.Scatter(x = df.index,
                                       y = df[l],
                                       mode = 'lines',
                                       marker = dict(color='green'),
                                      hovertemplate = hovertemplate,
                                      name='')
                            ],
                       layout = go.Layout()
                      )
    
    
    figure.update_layout(yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                     ),
                       title = dict(text = kunta+':<br>'+label+' päivittäin', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                         legend = dict(font=dict(size=18)),
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                       xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1,
                                         label="1kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=6,
                                         label="6kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="YTD",
                                         step="year",
                                         stepmode="todate"),
                                    dict(count=1,
                                         label="1v",
                                         step="year",
                                         stepmode="backward"),
                                    dict(step="all",label = 'MAX')
                                ])
                            ),
                            rangeslider=dict(
                                visible=True
                            ),
                            type="date"
                        )
                    )
    

    
    
    return figure

# Visualisoidaan haluttu muuttuja kuukausittain.
def plot_monthly_data(kunta, label):
    
    df = data[data.kunta_nimi==kunta]

    l = labels[label].replace('_kum','')
    
    df = df.resample('M')[l].sum()
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%B %Y'), '{:,}'.format(round(df.values[i],2)).replace(',',' ')) for i in range(len(df))]
    
    figure = go.Figure(data = [
                            go.Scatter(x = df.index,
                                       y = df.values,
                                      name = '',
                                       mode = 'lines+markers',
                                       marker = dict(color='purple'),
                                      hovertemplate = hovertemplate)
                            ],
                       layout = go.Layout()
                      )
    
    
    figure.update_layout(yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                     ),
                       title = dict(text = kunta+':<br>'+label+' kuukausittain', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                       legend = dict(font=dict(size=18)),
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                       xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1,
                                         label="1kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=6,
                                         label="6kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="YTD",
                                         step="year",
                                         stepmode="todate"),
                                    dict(count=1,
                                         label="1v",
                                         step="year",
                                         stepmode="backward"),
                                    dict(step="all",label = 'MAX')
                                ])
                            ),
                            rangeslider=dict(
                                visible=True
                            ),
                            type="date"
                        )
                    )
    

    
    
    return figure

# Visualisoidaan haluttu muuttuja kvartaaleittain.
def plot_quaterly_data(kunta, label):
    
    df = data[data.kunta_nimi==kunta]

    l = labels[label].replace('_kum','')
    
    df = df.resample('Q')[l].sum()
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(to_quartals(df.index[i]), '{:,}'.format(round(df.values[i],2)).replace(',',' ')) for i in range(len(df))]
    
    figure = go.Figure(data = [
                            go.Scatter(x = df.index,
                                       y = df.values,
                                      name = '',
                                       mode = 'lines+markers',
                                       marker = dict(color='blue'),
                                      hovertemplate = hovertemplate)
                            ],
                       layout = go.Layout()
                      )
    
    
    figure.update_layout(yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                     ),
                       title = dict(text = kunta+':<br>'+label+' kvartaaleittain', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                        xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                                                
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1,
                                         label="1kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=6,
                                         label="6kk",
                                         step="month",
                                         stepmode="backward"),
                                    dict(count=1,
                                         label="YTD",
                                         step="year",
                                         stepmode="todate"),
                                    dict(count=1,
                                         label="1v",
                                         step="year",
                                         stepmode="backward"),
                                    dict(step="all",label = 'MAX')
                                ])
                            ),
                            rangeslider=dict(
                                visible=True
                            ),
                            type="date"
                        )
                    )
    

    
    
    return figure
                    
# Visualisoidaan haluttu muuttuja kumulatiivisena.
def plot_cum_data(kunta, label):
    
    df = get_cum_city_data(kunta)
      
    l = labels[label]
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(df.iloc[i][l],2)).replace(',',' ')) for i in range(len(df))]
    
    figure = go.Figure(data = [
    
                            go.Scatter(x = df.index,
                                      y = df[l],
                                       name = '',
                                       mode = 'lines',
                                       marker = dict(color = 'red'),
                                       hovertemplate = hovertemplate
                                      )
                            ],
                            layout = go.Layout()
                      )
    
    
    
    
    
    figure.update_layout(
                       title = dict(text = kunta+':<br>'+label+' zrfdszszzzzzzzzzzzzzzzzzzzzqaaaaaaaaasskumulatiivisena', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',   
                          height = 600,
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                          xaxis = dict(title = dict(text='Aika',
                                                     font=dict(size=18, family = 'Arial Black')
                                                    ),
                                        tickfont = dict(size=14)
                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    )
                    )
    return figure    
    
# Visualisoidaan ennuste päivittäin.
def plot_daily_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    
    
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(daily_true.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(daily_true.values[i],2)).replace(',',' ')) for i in range(len(daily_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(daily_pred.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(daily_pred.values[i],2)).replace(',',' ')) for i in range(len(daily_pred))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = daily_true.index, 
                                   y = daily_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = daily_pred.index, 
                                   y = daily_pred.values, 
                                   name = 'Ennuste', 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red'))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' päivittäin', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    
    return figure
                       
# Visualisoidaan ennuste kvartaaleittain.                       
def plot_quaterly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
                       
    quaterly_true = daily_true.resample('Q').sum()
    quaterly_pred = daily_pred.resample('Q').sum()                    
                       
    quaterly_true.index = [to_quartals(i) for i in quaterly_true.index]
    quaterly_pred.index = [to_quartals(i) for i in quaterly_pred.index]
                       
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(quaterly_true.index[i], '{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' ')) for i in range(len(quaterly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(quaterly_pred.index[i], '{:,}'.format(round(quaterly_pred.values[i],2)).replace(',',' ')) for i in range(len(quaterly_pred))]
    
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = quaterly_true.index, 
                                   y = quaterly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = quaterly_pred.index, 
                                   y = quaterly_pred.values, 
                                   name = 'Ennuste', 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red'))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' kvartaaleittain', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure

# Visualisoidaan ennuste kuukausittain.
def plot_monthly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
                       
    monthly_true = daily_true.resample('M').sum()
    monthly_pred = daily_pred.resample('M').sum()                    
                       
    monthly_true.index = [i.strftime('%B %Y')  for i in monthly_true.index]
    monthly_pred.index = [i.strftime('%B %Y') for i in monthly_pred.index]
                       
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(monthly_true.index[i], '{:,}'.format(round(monthly_true.values[i],2)).replace(',',' ')) for i in range(len(monthly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(monthly_pred.index[i], '{:,}'.format(round(monthly_pred.values[i],2)).replace(',',' ')) for i in range(len(monthly_pred))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = monthly_true.index, 
                                   y = monthly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = monthly_pred.index, 
                                   y = monthly_pred.values, 
                                   name = 'Ennuste', 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red'))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    ),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          title = dict(text = kunta+':<br>'+label+' kuukausittain', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure

# Visualisoidaan ennuste kumulatiivisena.
def plot_cumulative_prediction(df):
    

    label_name = df.name.values[0]                    
    label = labels[label_name]
    kunta = df.kunta.values[0]
    
    df_true = df[df.forecast=='Toteutunut']
    df_pred = df[df.forecast=='Ennuste']
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(df_true.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(df_true.iloc[i][label],2)).replace(',',' ')) for i in range(len(df_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(df_pred.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(df_pred.iloc[i][label],2)).replace(',',' ')) for i in range(len(df_pred))]
    
    
    figure = go.Figure(data = [
                            
                            go.Scatter(x = df_true.index, 
                                       y = df_true[label], 
                                       name = 'Toteutunut', 
                                       hovertemplate = hover_true,
                                       marker = dict(color='green')),
                            go.Scatter(x = df_pred.index, 
                                       y = df_pred[label], 
                                       name = 'Ennuste', 
                                       hovertemplate = hover_pred,
                                       marker = dict(color='red'))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    ),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          title = dict(text = kunta+':<br>'+label_name+' kumulatiivisena', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure


# Visualisoidaan testitulokset päivittäin.
def plot_daily_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    test_data['prev'] = test_data.ennustettu.shift(periods=1)
    test_data.prev = test_data.prev.fillna(test_data.edellinen)
    daily_test = test_data['ennustettu'] - test_data['prev']
    daily_true = test_data[label] - test_data['edellinen']

    
    test_error = daily_test - daily_true
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / daily_true), 2)

    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(daily_test.index[i].strftime('%#d. %Bta %Y'),'{:,}'.format(round(daily_true[i],2)).replace(',',' '),'{:,}'.format(round(daily_test[i],2)).replace(',',' '),'{:,}'.format(round(test_error[i],2)).replace(',',' '),round(test_error_percentage[i],2)) for i in range(len(test_data))]
    

        
        
    figure = go.Figure(data=[

                go.Bar(x = daily_true.index, 
                       y = daily_true.values, 
                       name = 'Testidata',
                       hovertemplate = ['{}:<br>{} €'.format(daily_true.index[i].strftime('%#d. %Bta %Y'),
                                                          '{:,}'.format(round(daily_true.values[i],2)).replace(',',' ')) for i in range(len(daily_true))],
                       marker = dict(color='green')),
                go.Scatter(x = daily_test.index, 
                           y = daily_test.values, 
                           name = 'Ennuste',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red', size = 10)),
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14)
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' päivittäin',x=.5, font=dict(size=24,family = 'Arial')))
                      )
    
    return figure
    
# Visualisoidaan testitulokset kuukausittain.    
def plot_monthly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    test_data['prev'] = test_data.ennustettu.shift(periods=1)
    test_data.prev = test_data.prev.fillna(test_data.edellinen)
    daily_test = test_data['ennustettu'] - test_data['prev']
    daily_true = test_data[label] - test_data['edellinen']
    
    
    
    monthly_test = daily_test.resample('M').sum()
    monthly_true = daily_true.resample('M').sum()
       
      

    
    test_error = monthly_true - monthly_test
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / monthly_true), 2)


    monthly_true.index = [i.strftime('%B %Y')  for i in monthly_true.index]
    monthly_test.index = [i.strftime('%B %Y') for i in monthly_test.index]
    
    hover_true = ['{}:<br>{} €'.format(monthly_true.index[i],
                                                          '{:,}'.format(round(monthly_true.values[i],2)).replace(',',' ')) for i in range(len(monthly_true))]
    
    hover_test = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(monthly_true.index[i],'{:,}'.format(round(monthly_true.values[i],2)).replace(',',' '),'{:,}'.format(round(monthly_test.values[i],2)).replace(',',' '),'{:,}'.format(round(test_error.values[i],2)).replace(',',' '),round(test_error_percentage.values[i],2)) for i in range(len(monthly_true))]
    
    
    figure = go.Figure(data=[

                go.Bar(x = monthly_true.index, 
                       y = monthly_true.values, 
                       name = 'Testidata', 
                       hovertemplate = hover_true, 
                       marker = dict(color='green')),
                go.Scatter(x = monthly_test.index,
                           y = monthly_test.values, 
                           name = 'Ennuste',
                           mode = 'markers', 
                           hovertemplate=hover_test, 
                           marker = dict(color='red', size = 10)),
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14)
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kuukausittain',x=.5, font=dict(size=24,family = 'Arial'))
                                       )
                      ) 
    return figure
    
# Visualisoidaan testitulokset kvartaaleittain.
def plot_quaterly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    test_data['prev'] = test_data.ennustettu.shift(periods=1)
    test_data.prev = test_data.prev.fillna(test_data.edellinen)
    daily_test = test_data['ennustettu'] - test_data['prev']
    daily_true = test_data[label] - test_data['edellinen']
    
    quaterly_test = daily_test.resample('Q').sum()
    quaterly_true = daily_true.resample('Q').sum()
    

    
    test_error = quaterly_true - quaterly_test
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / quaterly_true), 2)

    
    
    quaterly_true.index = [to_quartals(i) for i in quaterly_true.index]
    quaterly_test.index = [to_quartals(i) for i in quaterly_test.index]
    
    
    hover_true = ['{}:<br>{} €'.format(quaterly_true.index[i],'{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' ')) for i in range(len(quaterly_true))]
    
    hover_test = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(quaterly_true.index[i],'{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' '),'{:,}'.format(round(quaterly_test.values[i],2)).replace(',',' '),'{:,}'.format(round(test_error.values[i],2)).replace(',',' '),round(test_error_percentage.values[i],2)) for i in range(len(quaterly_true))]
        
    figure = go.Figure(data=[

                go.Bar(x = quaterly_true.index, 
                       y = quaterly_true.values, 
                       name = 'Testidata',
                       hovertemplate=hover_true, 
                       marker = dict(color='green')),
                go.Scatter(x = quaterly_test.index, 
                           y = quaterly_test.values, 
                           name = 'Ennuste',
                           mode = 'markers', 
                           hovertemplate=hover_test,
                           marker = dict(color='red', size = 10)),
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14)
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kuukausittain',x=.5, font=dict(size=24,family = 'Arial'))
                                       )
                      )
    return figure
    
# Visualisoidaan testitulokset kumulatiivisena.    
def plot_cumulative_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    
    kunta = df.kunta.values[0]
    
    test_data = df[df.split=='test']
    val_data = df[df.split=='val']
    train_data = df[df.split=='train']

    test_error = test_data.ennustettu - test_data[label]
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / test_data[label]), 2)

    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(test_data.index[i].strftime('%#d. %Bta %Y'),'{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' '),'{:,}'.format(round(test_data.iloc[i]['ennustettu'],2)).replace(',',' '),'{:,}'.format(round(test_error[i],2)).replace(',',' '),round(test_error_percentage[i],2)) for i in range(len(test_data))]
    
    
    figure = go.Figure(data=[

                go.Scatter(x = train_data.index, 
                           y = train_data[label], 
                           name = 'Opetusdata', 
                           hovertemplate = ['{}:<br>{} €'.format(train_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(train_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(train_data))],
                           marker = dict(color = 'purple')),
                go.Scatter(x = val_data.index, 
                           y = val_data[label], 
                           name = 'Validointidata', 
                           hovertemplate = ['{}:<br>{} €'.format(val_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(val_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(val_data))],
                           marker = dict(color = 'orange')),

                go.Scatter(x = test_data.index, 
                           y = test_data[label], 
                           name = 'Testidata', 
                           hovertemplate = ['{}:<br>{} €'.format(test_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(test_data))],
                           marker = dict(color='green')),
                go.Scatter(x = test_data.index, 
                           y = test_data.ennustettu, 
                           name = 'Ennuste', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red'))

            ],
                           layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                  font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14)
                                                    ),
                                            legend = dict(font=dict(size=18)),
                                            height = 600,
                                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name,x=.5, font=dict(size=24,family = 'Arial'))
                                           )
                      )
    
    return figure
    

def serve_layout():
    
    return html.Div(children = [
        
               html.Br(),
               html.H1('Perustoimeentulotuki Suomessa',style={'textAlign':'center', 'font-size':60, 'color':'black'}),
               html.Br(),
               html.H2('ja sen ennakointi kunnittain',style={'textAlign':'center', 'font-size':40, 'color':'black'}),
               html.Br(),
              
              
                dbc.Row(children = [
                    
                    dbc.Col(children=[
                       html.Br(),
                       html.H3('Valitse kunta.',style={'textAlign':'center', 'color':'black'}),
                       html.Br(),
                       dcc.Dropdown(id = 'kunta_dropdown', 
                            options = kunta_options, 
                            value = 'Turku',
                            multi = False,
                            style = {'font-size':18, 'font-family':'Arial','color': 'black'},
                            placeholder = 'Valitse kunta.'),

                       html.Br()

                    ],xs =10, sm=8, md=5, lg=4, xl=4, align = 'center'),
                
                  
                   
                   dbc.Col(children=[
                       html.Br(),
                       html.Br(),
                       html.H3('Valitse muuttuja.',style={'textAlign':'center', 'color':'black'}),
                       html.Br(),
                       dcc.Dropdown(id = 'label_dropdown', 
                            options = label_options, 
                            value = 'Maksut yhteensä',
                            multi = False,
                            style = {'font-size':18, 'font-family':'Arial','color': 'black'},
                            placeholder = 'Valitse muuttuja.'),


                       html.Br(),

                       html.Br(),
                   ],xs =10, sm=8, md=5, lg=8, xl=8)
                ], style = {'margin' : '10px 10px 10px 10px'}),
        
        
        dbc.Row(children = [
        
            dbc.Col(id = 'original_graph_col',xs =10, sm=8, md=5, lg=5, xl=5, align = 'center'),
            

                        dbc.Col([dbc.RadioItems(id = 'orig_resampler', 
                                      options = [{'label':'Päivittäin','value':'D'}, 
                                                 {'label':'Kuukausittain', 'value':'M'}, 
                                                 {'label':'Kvartaaleittain', 'value': 'Q',}
                                          
                                                ],
                                      className="form-group",
                                      inputClassName="form-check",
                                      labelClassName="btn btn-outline-warning",
                                      labelCheckedClassName="active",
                                      
                                      value = 'D',
                                     labelStyle={'font-size':22, 'display':'block'}
                                     )
                                ],
                                style = {'textAlign':'center'},
                               xs =10, sm=8, md=5, lg=2, xl=2, align = 'center'
                               ),

            
              dbc.Col(id = 'cumulative_graph_col',xs =10, sm=8, md=5, lg=5, xl=5, align = 'center')
        
        
        
        ], style = {'margin' : '10px 10px 10px 10px'}),
        
        
        dbc.Row(children = [
        
            dbc.Col(children = [
            
                       html.H3('Valitse testi -ja validointidatan osuus.',style={'textAlign':'center', 'color':'black'}),
                       dcc.Slider(id = 'test_slider',
                                 min = .1,
                                 max = .5,
                                 value = .3,
                                 step = .01,
                                 marks = {.1: {'label':'10 %', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          .3:{'label':'30 %', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          .5:{'label':'50 %', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}}

                                          }
                                 ),
                        
                      html.Div(id = 'test_size_indicator', 
                               children = [html.P('Valitsit {} prosentin testiosuuden.'.format(30),style = {'textAlign':'center', 'fontSize':24, 'fontFamily':'Arial Black', 'color':'black'})])
            
            
            
            
            ],xs =10, sm=8, md=5, lg=5, xl=5, align = 'center'),
            dbc.Col(xs =10, sm=8, md=5, lg=1, xl=1, align = 'center'),
            dbc.Col(children = [
                       html.Br(),
                       html.H3('Valitse ennusteen pituus.',style={'textAlign':'center', 'color':'black'}),
                       dcc.Slider(id = 'forecast_slider',
                                 min = 30,
                                 max = 4*365,
                                 value = 90,
                                 step = 1,
                                 marks = {30: {'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                        #  180:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          365:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          2*365:{'label':' kaksi vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          4*365:{'label':'neljä vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}}

                                          }
                                 ),
                       html.Br(),
                       html.Div(id = 'forecast_slider_indicator', 
                                children = [html.P('Valitsit {} päivän ennusteen.'.format(90),style = {'textAlign':'center', 'fontSize':24, 'fontFamily':'Arial Black', 'color':'black'})])
            
            
            
            
            ],xs =10, sm=8, md=5, lg=6, xl=6, align = 'center')
        
        
        ], style = {'margin' : '10px 10px 10px 10px'}),


       
        dbc.Row([ dbc.Col(xs =10, sm=8, md=5, lg=5, xl=5, align = 'center'),
                  dbc.Col(     dbc.Button('Testaa ja ennusta',
                                  id='start_button',
                                  n_clicks=0,
                                  outline=False,
                                  className="btn btn-outline-info",
                                  style = dict(fontSize=36)
                                  ),xs =10, sm=8, md=5, lg=2, xl=2, align = 'start'),
                  dbc.Col(xs =10, sm=8, md=5, lg=5, xl=5, align = 'center')  

                        
                    
                     
                    ],justify='center', style = {'margin' : '10px 10px 10px 10px'}),
        
        html.Br(),
        
        dbc.Tabs(children= [
            
            dbc.Tab(label = 'Testi ja ennuste',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28}, 
                    children = [
       
                        html.Br(),
                        dcc.Loading(html.Div(id = 'results', style = {'margin' : '10px 10px 10px 10px'}),
                                   type = spinners[random.randint(0,len(spinners)-1)]),
                        html.Div(id = 'hidden_data_div',
                                 children= [
                                            dcc.Store(id='predict_store'),
                                            dcc.Store(id='test_store'),
                                            dcc.Download(id = "download-component")
                                           ]
                                 )
                    ]
                    

                                   ),
            dbc.Tab(label = 'Ohje',
                   tabClassName="flex-grow-1 text-center",                    
                   tab_style = {'font-size':28},
                   children = [ 
                       dbc.Row(justify='center', children=[
                       
                           dbc.Col(xs =10, sm=8, md=5, lg=6, xl=6, children =[
                   

                                html.Br(),
                                html.H4('Johdanto',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P('Tämä sovellus hyödyntää Kelan tilastoimia päivittäistä dataa toimeentulotukien maksuista ja palautuksista ja pyrkii muodostamaan koneoppimisen avulla ennusteen tulevaisuuteen. Käyttäjä voi valita haluamansa tukilajin sekä kunnan alasvetovalikoista, tarkastella toteumadataa sekä tehdä valitun pituisen ennusteen.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('Sovellus hyödyntää lineaarista regressiota Ridge-regularisoinnilla. Ohjelma optimoi algoritmin regularisointiparametrin ja suorittaa lopullisen ennusteen parhaalla mahdollisella algoritmilla.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('Ennusteen laatua pystyy tarkastelemaan vertailemalla toteutunutta dataa sekä testissä tehtyä ennustetta. Näin käyttäjä saa parempaa tietoa ennusteen luotettavuudesta.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.H4('Ohje',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P('1. Valitse haluttu kunta.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('2. Valitse haluttu suure. Voit tarkastella suuretta eri ajanjaksoissa tai kumulatiivisesti.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),             
                                html.Br(),
                                html.P('3. Valitse testidatan osuus. Tästä puolet käytetään validointidatana. Validointidatan avulla koneoppimisalgoritmi valitsee hyperparametrit.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('4. Valitse ennusteen pituus. Lopullinen algoritmi laskee ennusteen halutulle ajalle.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('5. Klikkaa "testaa ja ennusta". Tämän jälkeen voit tarkastella testin tuloksia sekä tehtyä ennustetta halutulla ajanjaksolla.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.H4('Vastuuvapauslauseke',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P("Sivun ja sen sisältö tarjotaan ilmaiseksi sekä sellaisena kuin se on saatavilla. Kyseessä on yksityishenkilön tarjoama palvelu eikä viranomaispalvelu. Sivulta saatavan informaation hyödyntäminen on päätöksiä tekevien tahojen omalla vastuulla. Palvelun tarjoaja ei ole vastuussa menetyksistä, oikeudenkäynneistä, vaateista, kanteista, vaatimuksista, tai kustannuksista taikka vahingosta, olivat ne mitä tahansa tai aiheutuivat ne sitten miten tahansa, jotka johtuvat joko suoraan tai välillisesti yhteydestä palvelun käytöstä. Huomioi, että tämä sivu on yhä kehityksen alla.",style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.H4('Tuetut selaimet',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P("Sovellus on testattu toimivaksi Google Chromella ja Mozilla Firefoxilla. Edge- ja Internet Explorer -selaimissa sovellus ei toimi. Opera, Safari -ja muita selaimia ei ole testattu.",style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.Div(style={'text-align':'center'},children = [
                                           html.H4('Lähteet', style = {'text-align':'center', 'color':'black'}),
                                           html.Br(),
                                           html.Label(['Tilastokeskus: ', 
                                                    html.A('Kelan maksaman perustoimeentulotuen menot ja palautukset', href = "https://www.avoindata.fi/data/fi/dataset/kelan-maksaman-perustoimeentulotuen-menot-ja-palautukset",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.Br(),
                                           html.Label(['Wikipedia: ', 
                                                    html.A('Ridge-regressio (englanniksi)', href = "https://en.wikipedia.org/wiki/Ridge_regression",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br()

                                       ]),
                                 html.Br(),
                                 html.H4('Tekijä', style = {'text-align':'center','color':'black'}),
                                       html.Br(),
                                       html.Div(style = {'text-align':'center'},children = [
                                           html.I('Tuomas Poukkula', style = {'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.A('Seuraa LinkedIn:ssä', href='https://www.linkedin.com/in/tuomaspoukkula/', target = '_blank',style = {'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.A('tai Twitterissä.', href='https://twitter.com/TuomasPoukkula', target = '_blank',style = {'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.Br(),
                                           html.Label(['Sovellus ', 
                                                    html.A('GitHub:ssa', href='https://github.com/tuopouk/suomenavainklusterit')
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
                                       ])



                            ]
                       
                              )
                   ]

                           )
                   ]
                   )
                ]
                        )
            ], style = {'margin' : '10px 10px 10px 10px'}
                           )




        



@app.callback(
    Output('test_size_indicator','children'),
    [Input('test_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.P('Valitsit {} prosentin testiosuuden.'.format(int(round(100*value,1))),style = {'textAlign':'center', 'fontSize':24, 'fontFamily':'Arial Black', 'color':'black'})]

@app.callback(
    Output('forecast_slider_indicator','children'),
    [Input('forecast_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.P('Valitsit {} päivän ennusteen.'.format(value),style = {'textAlign':'center', 'fontSize':24, 'fontFamily':'Arial Black', 'color':'black'})]

@app.callback(
    [Output('results','children'),
    ServersideOutput('predict_store','data'),
    ServersideOutput('test_store','data')],
    [Input('start_button', 'n_clicks'),
    State('kunta_dropdown','value'),
    State('label_dropdown', 'value'),
    State('test_slider', 'value'),
    State('forecast_slider','value')]
)
def start(n_clicks, kunta, label_name, test, length):
    
    if n_clicks > 0:
    
        label = labels[label_name]

        dataset = shift_dates(dataset = get_cum_city_data(kunta), label = label)

        train_val_test_df = train_val_test(dataset, label, test)
        
        train_val_test_df['daily_true'] = train_val_test_df[label] - train_val_test_df['edellinen']
        train_val_test_df['daily_pred'] = train_val_test_df['ennustettu'] - train_val_test_df['edellinen']
        train_val_test_df['name'] = label_name
        train_val_test_df['kunta'] = kunta

        alpha = train_val_test_df.alpha.dropna().values[0]

        prediction = predict(dataset, label, length, alpha)
        
        prediction['daily'] = prediction[label] - prediction['edellinen']
        prediction['name'] = label_name
        prediction['kunta'] = kunta
        
        true_df = prediction[prediction.forecast=='Toteutunut']
        pred_df = prediction[prediction.forecast=='Ennuste']
        
        
        hover_true = ['<b>{}</b>:<br>{} €'.format(true_df.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(true_df.iloc[i][label],2)).replace(',',' ')) for i in range(len(true_df))]
    
        hover_pred = ['<b>{}</b>:<br>{} €'.format(pred_df.index[i].strftime('%#d. %Bta %Y'), '{:,}'.format(round(pred_df.iloc[i][label],2)).replace(',',' ')) for i in range(len(pred_df))]
    


        predict_fig = go.Figure(data =[

            go.Scatter(x = true_df.index, 
                       y = true_df[label], 
                       name = 'Toteutunut', 
                       hovertemplate = hover_true,
                       marker = dict(color='green')),
            go.Scatter(x = pred_df.index, 
                       y = pred_df[label], 
                       name = 'Ennuste', 
                       hovertemplate = hover_pred,
                       marker = dict(color='red'))

        ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)), 
                                          yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                   font=dict(size=16, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                      ),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                         title = dict(text = kunta+':<br>'+label_name+' kumulatiivisena', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                               )


        test_data = train_val_test_df[train_val_test_df.split=='test']
        val_data = train_val_test_df[train_val_test_df.split=='val']
        train_data = train_val_test_df[train_val_test_df.split=='train']

        test_error = test_data.ennustettu - test_data[label]
        test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / test_data[label]), 2)

        hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(test_data.index[i].strftime('%#d. %Bta %Y'),'{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' '),'{:,}'.format(round(test_data.iloc[i]['ennustettu'],2)).replace(',',' '),'{:,}'.format(round(test_error[i],2)).replace(',',' '),round(test_error_percentage[i],2)) for i in range(len(test_data))]

        train_val_test_fig = go.Figure(data=[

                go.Scatter(x = train_data.index, 
                           y = train_data[label], 
                           name = 'Opetusdata',
                           hovertemplate = ['{}:<br>{} €'.format(train_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(train_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(train_data))],
                           marker = dict(color = 'purple')),
                go.Scatter(x = val_data.index, 
                           y = val_data[label], 
                           name = 'Validointidata', 
                           hovertemplate = ['{}:<br>{} €'.format(val_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(val_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(val_data))],
                           marker = dict(color = 'orange')),

                go.Scatter(x = test_data.index, 
                           y = test_data[label], 
                           name = 'Testidata', 
                           hovertemplate = ['{}:<br>{} €'.format(test_data.index[i].strftime('%#d. %Bta %Y'),
                                                             '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(test_data))],
                           marker = dict(color='green')),
                 
                go.Scatter(x = test_data.index, 
                           y = test_data.ennustettu, 
                           name = 'Ennuste',
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red')),

            ],
                           layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                     ),
                                                         tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text=label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14)
                                                      ),
                                            legend = dict(font=dict(size=18)),
                                            height = 600,
                                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name,x=.5, font=dict(size=24,family = 'Arial'))))
        

        return [dbc.Row(justify='center',children=[
                    dbc.Col([html.Br(),
                             html.Br(),
                             dbc.Card(dcc.Graph(id = 'train_val_test_fig', 
                                                figure =train_val_test_fig),
                                      color='dark',body=True),
                             html.P('λ = '+str(alpha)),
                             html.Br(),
                             html.P('Tällä kuvaajalla voit tarkastella ennustettua muuttujaa joko päivittäin, kuukausittain, kvartaaleittain tai kumulatiivisena. Kuvaajassa vihreällä värillä on esitetty testidata sekä punaisella testissä tehty ennuste. Kumulatiivisessa kuvaajassa on piirretty myös opetus -ja validointidata, joita on hyödynnety regressiomallin opetuksessa sekä algoritmin optimoinnissa. Kuvaaja näyttää myös testin ja toteutuneen datan välisen eron. Yleisesti ottaen virhe on suurin päivittäisissä ennusteissa ja pienin kumulatiivisessa ennusteessa.', style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})],
                            xs =10, sm=8, md=5, lg=5, xl=5, align = 'center'),
                
                   dbc.Col([dbc.RadioItems(id = 'resampler', 
                                      options = [{'label':'Päivittäin','value':'D'}, 
                                                 {'label':'Kuukausittain', 'value':'M'}, 
                                                 {'label':'Kvartaaleittain', 'value': 'Q',},
                                                 {'label':'Kumulatiivisena', 'value': 'KUM'}
                                                ],
                                      className="form-group",
                                      inputClassName="form-check",
                                      labelClassName="btn btn-outline-warning",
                                      labelCheckedClassName="active",
                                      
                                      value = 'KUM',
                                     labelStyle={'font-size':22, 'display':'block'}
                                     )
                                ],
                                style = {'textAlign':'center'},
                               xs =10, sm=8, md=5, lg=2, xl=2, align = 'center'
                               ),
                
               
                
                
                   dbc.Col([dbc.Card(dcc.Graph(id = 'predict_fig', 
                                               figure =predict_fig),
                                     color='dark',
                                     body=True),
                           html.Br(),
                           html.P('Tässä kuvaajassa esitetään itse ennuste, jota voi tarkastella, testitulosten tavoin, joko päivittäin, kuukausittain, kvartaaleittain tai kumulatiivisena. Yleisesti ottaen, ennuste on kumulatiivisessa muodossaan tarkimmillaan ja heikoimillaan päiväkohtaisessa ennusteessa.', style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
                           ],
                           xs =10, sm=8, md=5, lg=5, xl=5, align = 'center')
               ]),
                dbc.Row(justify='center',children=[
                    
                    dbc.Col(xs =10, sm=8, md=5, lg=5, xl=5, align = 'center'),
                    dbc.Col([
                           dbc.Button(children=[
                               html.I(className="fa fa-download mr-1"), ' Lataa tiedosto koneelle'],
                                   id='download_button',
                                   n_clicks=0,
                                   outline=True,
                                   size = 'lg',
                                   color = 'light'
                                   )
                            ],xs =10, sm=8, md=5, lg=2, xl=2, align = 'center'),
                    dbc.Col(xs =10, sm=8, md=5, lg=5, xl=5, align = 'center')
                        ])
               ],prediction,train_val_test_df
                
    
    

    
@app.callback(

    Output('predict_fig', 'figure'),
    [Input('predict_store', 'data'),
    Input('resampler', 'value')]

)
def update_predict_graph(df, rs):
    
    if rs is None or df is None:
        raise PreventUpdate
    
    if rs == 'D':
        return plot_daily_prediction(df)
    elif rs == 'M':
        return plot_monthly_prediction(df)
    elif rs == 'Q':
        return plot_quaterly_prediction(df)
    else:
        return plot_cumulative_prediction(df)
    
    
    
@app.callback(

    Output('train_val_test_fig', 'figure'),
    [Input('test_store', 'data'),
    Input('resampler', 'value')]

)
def update_test_graph(df, rs):
    
    if rs is None or df is None:
        raise PreventUpdate
    
    if rs == 'D':
        return plot_daily_test(df)
    elif rs == 'M':
        return plot_monthly_test(df)
    elif rs == 'Q':
        return plot_quaterly_test(df)
    else:
        return plot_cumulative_test(df)    
    
    
    
@app.callback(

    Output('original_graph_col', 'children'),
    [Input('kunta_dropdown', 'value'),
     Input('orig_resampler', 'value'),
    Input('label_dropdown', 'value')]

)
def update_daily_graph(kunta, rs, label):
    
    if rs is None or kunta is None:
        raise PreventUpdate
    
    if rs == 'D':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                figure = plot_daily_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa päivittäin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'M':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                figure = plot_monthly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa kuukausittain.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'Q':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                figure = plot_quaterly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa kvartaaleittain.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
     
    




@app.callback(

    Output('cumulative_graph_col', 'children'),
    
    [Input('kunta_dropdown', 'value'),
     Input('label_dropdown', 'value')
    ]

)
def update_cum_col(kunta, label):
    
    figure = plot_cum_data(kunta, label)
    
    return html.Div([dbc.Card(dcc.Graph(id = 'cumulative_graph',
                                        figure = figure),
                              color='dark',
                              body=True),
                    html.Br(),
                    html.P('Tässä kuvaajassa voit tarkastella valitun muuttujan kumulatiivista kehitystä.',
                          style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
                    ])


        
@app.callback(
    Output("download-component", "data"),
    [Input("download_button", "n_clicks"),
     State('predict_store','data'),
     State('test_store','data')
    ]
    
)
def download(n_clicks, prediction, train_val_test):
    
    if n_clicks > 0:
        

        label = prediction.name.values[0]
        kunta = prediction.kunta.values[0]
        pituus = prediction.ennusteen_pituus.values[0]
        pr = prediction.copy()
        pr.drop(['name','kunta','ennusteen_pituus'],axis=1, inplace = True)
        pr = pr.rename(columns={'forecast':'Ennuste/Toteutunut',
                               'daily':'arvo_pv'})
        
        
        train_data = train_val_test[train_val_test.split=='train'].copy()
        val_data = train_val_test[train_val_test.split=='val'].copy()
        test_data = train_val_test[train_val_test.split=='test'].copy()
        
        alpha = test_data.alpha.values[0]
        split_portion = test_data.split_portion.values[0]
        
        train_data.drop(['split','alpha','name','kunta', 'ennustettu', 'daily_pred','split_portion'],axis=1, inplace=True)
        val_data.drop(['split', 'alpha','name','kunta', 'ennustettu', 'daily_pred','split_portion'],axis=1, inplace=True)
        test_data.drop(['split', 'alpha', 'name', 'kunta','split_portion'],axis=1, inplace=True)
        
        train_data = train_data.rename(columns = {'daily_true':'arvo_pv'})
        val_data = val_data.rename(columns = {'daily_true':'arvo_pv'})
        test_data = test_data.rename(columns = {'daily_true':'arvo_pv', 'daily_pred':'ennuste_pv'})
        
        metadata = pd.DataFrame([{'Kunta':kunta,
                                 'Suure': label,
                                 'Testi/validointiosuus': str(int(100*split_portion))+'%',
                                 'Regularisointisuure':alpha,
                                 'Ennusteen pituus': str(pituus)+' päivää'}]).T.reset_index()
        metadata.columns = ['Tieto','Arvo']
        metadata = metadata.set_index('Tieto')
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        metadata.to_excel(writer, sheet_name= 'Metadata')
        train_data.to_excel(writer, sheet_name= 'Opetusdata')
        val_data.to_excel(writer, sheet_name= 'Validointidata')
        test_data.to_excel(writer, sheet_name= 'Testidata')
        
        pr.to_excel(writer, sheet_name= 'Ennustedata')
        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, kunta+'_'+label+'_'+datetime.now().strftime('%d_%m_%Y')+'.xlsx')



app.layout = serve_layout
if __name__ == "__main__":
    app.run_server(debug=False)