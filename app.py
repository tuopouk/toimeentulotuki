#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
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
import time
import locale
import warnings
warnings.filterwarnings("ignore")
locale.setlocale(locale.LC_ALL, 'fi_FI')

pd.set_option('use_inf_as_na', True)



# Käänteisen etäisyyden normalisointi.
distance_baseline = .75

# Onko kehitysversio?
in_dev = False

# Kuinka monta sekuntia saa metodi kestää. Tarvitaan herokua varten.
heroku_threshold = {True:10*60, False:20}[in_dev]

spinners = ['graph', 'cube', 'circle', 'dot' ,'default']

config_plots = {"locale":"fi",
                "modeBarButtonsToRemove":["sendDataToCloud"],
               "displaylogo":False}

features = ['edellinen', 


    'first_pay_day_distance',
    'second_pay_day_distance',
    'third_pay_day_distance',
       'fourth_pay_day_distance']


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
           external_scripts = ["https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js"],
          # meta_tags = [{'name':'viewport',
                   #     'content':'width=device-width, initial_scale=1.0, maximum_scale=1.2, minimum_scale=0.5'}],
           external_stylesheets = external_stylesheets
          )


app.title = 'Toimeentulotuki Suomessa'

app.scripts.append_script({"external_url": "https://cdn.plot.ly/plotly-locale-fi-latest.js"})

# Haetaan Kelan toimeentulotukidata.
def get_kela_data():
    
    kela = pd.read_csv('https://www.avoindata.fi/data/dataset/4b64be55-5a69-4f6b-a9d5-d4cbbe5c4382/resource/6409bdec-4a48-46a9-8729-5a727e37cd55/download/data.csv')

    
    kela.maksu_pv = pd.to_datetime(kela.maksu_pv)
    kela = kela.set_index('maksu_pv')
    kela = kela.sort_index()
    kela.drop(['vuosikuukausi','vuosi','kunta_nro','etuus'],axis=1,inplace=True)
    
    koko_maa = kela.copy()
    koko_maa = koko_maa.reset_index().groupby(['maksu_pv', 'palautus']).sum()
    koko_maa = koko_maa.reset_index().set_index('maksu_pv')
    koko_maa['kunta_nimi'] = 'Koko maa'
    
    kela = pd.concat([kela,koko_maa])
    kela = kela.sort_index()
    
    return kela

keladata = get_kela_data()

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

data = get_combined_data(keladata)

"""
Maksupäivät ovat kuukauden 1., 9., 16. ja 23. Ensimmäinen osa maksetaan aina kuun ensimmäisenä pankkipäivänä. Muiden erien maksupäivää aikaistetaan, jos niiden maksupäivä osuu viikonloppuun tai pyhäpäivään.

Alla on funktioita, joilla lasketaan käänteinen etäisyys seuraavaan maksupäivään. Etäisyys ilmaisee kuinka monta päivää on nykyisen sekä
maksupäivän välillä. Käänteinen etäisyys on siten etäisyyden käänteisluku. Määrettä on normalisoitu siten, että etäisyyden ollessa nolla,
käänteinen etäisyys on yksi, ja etäisyyden ollessa 1, kännteinen etäiysyys on 0.75. Näin vältetään nollalla jako sekä samat arvot etäisyyden
ollessa yksi tai nolla.

"""



def get_holidays(years):
    return [k.strftime('%Y-%m-%d') for k in holidays.FIN(years=years).keys()]


def next_first_bdate(date):
    
    date = pd.to_datetime(date)
    target = pd.to_datetime('{}-{}-01'.format(date.year,date.month))
    if date > target:
        target += pd.DateOffset(months=1)
    return pd.to_datetime(np.busday_offset(target.strftime('%Y-%m-%d'), 0, roll='forward', holidays = get_holidays([target.year])))


def next_second_bdate(date):
    
    date = pd.to_datetime(date)
    target = pd.to_datetime('{}-{}-09'.format(date.year,date.month))
    if date > target:
        target += pd.DateOffset(months=1)
    return pd.to_datetime(np.busday_offset(target.strftime('%Y-%m-%d'), 0, roll='preceding', holidays = get_holidays([target.year])))
    
def next_third_bdate(date):
    
    date = pd.to_datetime(date)
    target = pd.to_datetime('{}-{}-16'.format(date.year,date.month))
    if date > target:
        target += pd.DateOffset(months=1)
    return pd.to_datetime(np.busday_offset(target.strftime('%Y-%m-%d'), 0, roll='preceding', holidays = get_holidays([target.year])))

def next_fourth_bdate(date):
    
    date = pd.to_datetime(date)
    target = pd.to_datetime('{}-{}-23'.format(date.year,date.month))
    if date > target:
        target += pd.DateOffset(months=1)
    return pd.to_datetime(np.busday_offset(target.strftime('%Y-%m-%d'), 0, roll='preceding', holidays = get_holidays([target.year])))

def bdates_in_between(date1,date2):
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    return np.busday_count(date1.strftime('%Y-%m-%d'),date2.strftime('%Y-%m-%d'), holidays = get_holidays([date1.year,date2.year]))

def inverse_distance(date1, date2):
    
    distance = bdates_in_between(date1,date2)
    
    return {True: 1.0, False: distance_baseline / distance}[distance == 0]

def first_pay_day_distance(date):
    
    return inverse_distance(date, next_first_bdate(date))

def second_pay_day_distance(date):
    
    return inverse_distance(date, next_second_bdate(date))

def third_pay_day_distance(date):
    
    return inverse_distance(date, next_third_bdate(date))

def fourth_pay_day_distance(date):
    
    return inverse_distance(date, next_fourth_bdate(date))


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
    df['first_pay_day_distance'] = df.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
    df['second_pay_day_distance'] = df.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
    df['third_pay_day_distance'] = df.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
    df['fourth_pay_day_distance'] = df.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values
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

## Tavallinen OLS -lineaariregressio.
def baseline(train_data, test_data, label):
    
    single_feature = ['edellinen']
    
    model = LinearRegression(n_jobs=-1)

    scl = StandardScaler()
    
    x_train = train_data[single_feature]
    
    X_train = scl.fit_transform(x_train)
    
    y_train = train_data[label]
    
    model.fit(X_train, y_train)
    
    
    df = train_data.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan    
    next_day = test_data.index.values[0]
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')
    
    df[label] = np.maximum(df.edellinen,model.predict(scl.transform(df[single_feature])))

    dfs = []
    dfs.append(df)

    days = len(test_data)
    
    for i in tqdm(range(1,days)):

        dff = dfs[-1].copy()

        dff.edellinen = dff[label]
        dff[label] = np.nan
        next_day = test_data.index.values[i]
        
        
        dff['maksu_pv'] = next_day
        dff = dff.set_index('maksu_pv')


        dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[single_feature])))
        dfs.append(dff)
        
    test_data = pd.concat(dfs)
    
    return test_data[['edellinen',label]]
    
    


# Jaetaan data opetus, validointi ja testidataan.
# Optimoidaan hyperparametri validointidatalla
# ja testataan lopullinen algoritmi testidatalla.
# Palautetaan tulosmatriisi.
def train_val_test(dataset, label, reg_type, test_size = 100):
    
    # Train - test -jako
    
    #train_size = 1 - test_size
    
    train_size = len(dataset) - test_size
    
    train_data = dataset.iloc[:train_size,:]
    test_data = dataset.iloc[train_size:,:]
    val_data = test_data.iloc[:int(len(test_data)/2),:]
    test_data = test_data.iloc[int(len(test_data)/2):,:]
    
#     train_data = dataset.iloc[:int(train_size*len(dataset)),:]
#     test_data = dataset.iloc[int(train_size*len(dataset)):,:]
#     val_data = test_data.iloc[:int(len(test_data)/2),:]
#     test_data = test_data.iloc[int(len(test_data)/2):,:]
    
    df = train_data.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan
    next_day = val_data.index.values[0]
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')
    df['first_pay_day_distance'] = df.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
    df['second_pay_day_distance'] = df.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
    df['third_pay_day_distance'] = df.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
    df['fourth_pay_day_distance'] = df.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values

    
    alpha_list = []
    
    # Alpha saa arvoja väliltä [2**-10, 2**10].
    # Alphaa sanotaan useimmin lambdaksi. Scikit-learnissa se on kuitenkin alpha.
    
    regularization_params = [2**c for c in range(-10,11)]
    #regularization_params.append(0)
    
    start = time.time()
    end = time.time()
    
    
    for alpha in tqdm(regularization_params):
        
        # Jos on kestänyt alle threshold-arvon, jatka.
        if end - start < heroku_threshold:
            model = {'Lasso':Lasso(random_state=42, alpha = alpha),
                 'Ridge': Ridge(random_state=42, alpha = alpha),
                 'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)}[reg_type]
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
                dff['first_pay_day_distance'] = dff.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
                dff['second_pay_day_distance'] = dff.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
                dff['third_pay_day_distance'] = dff.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
                dff['fourth_pay_day_distance'] = dff.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values


                dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features])))
                dfs.append(dff)
            error = np.absolute(val_data.iloc[-1][label]-pd.concat(dfs).iloc[-1][label])
            alpha_list.append({'alpha':alpha, 'error':error})
            
            end = time.time()
        else:
            break
            
            
      
    alpha = pd.DataFrame(alpha_list).sort_values(by='error').head(1).alpha.values[0]
    
    train_data_prev = train_data.copy()
    
    train_data = pd.concat([train_data,val_data])
                    
    #model = Lasso(random_state=42, alpha = alpha)
    model = {'Lasso':Lasso(random_state=42, alpha = alpha),
                 'Ridge': Ridge(random_state=42, alpha = alpha),
                 'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)}[reg_type]
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
    df['first_pay_day_distance'] = df.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
    df['second_pay_day_distance'] = df.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
    df['third_pay_day_distance'] = df.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
    df['fourth_pay_day_distance'] = df.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values

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
        dff['first_pay_day_distance'] = dff.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
        dff['second_pay_day_distance'] = dff.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
        dff['third_pay_day_distance'] = dff.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
        dff['fourth_pay_day_distance'] = dff.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values

        dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features])))
        dfs.append(dff)
    
    testi_df = pd.concat(dfs)
    
    test_data['ennustettu'] = testi_df[label]
    test_data['ennuste_edellinen'] = testi_df['edellinen']
    
    baseline_df = baseline(train_data, test_data, label)
    
    test_data['baseline'] = baseline_df[label]
    test_data['baseline_edellinen'] = baseline_df['edellinen']
    
    train_data_prev['split'] = 'train'
    val_data['split'] = 'val'
    test_data['split'] = 'test'   
  
        
    result = pd.concat([train_data_prev, val_data, test_data])
    result['alpha'] = alpha
    result['split_portion'] = test_size
    result['reg_type'] = {'Lasso':'Lasso','Ridge':'Ridge', 'ElasticNet': 'Elastinen verkko'}[reg_type]
    result['label'] = label
   
   

    return result

# Tuotetaan ennuste halutulle ajalle valitulla alpha-parametrilla.
# Voidaan tuottaa tavallinen lineaariregressio, baseline -muuttujan ollessa True.
def predict(dataset, label, length, alpha, reg_type, baseline = False):
    
    if baseline:
        
        model = LinearRegression(n_jobs = -1)
        features_ = ['edellinen']
    else:
        
        model = {'Lasso':Lasso(random_state=42, alpha = alpha),
                 'Ridge': Ridge(random_state=42, alpha = alpha),
                 'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)}[reg_type]
    

        features_ = features
        

    scl = StandardScaler()
    x = dataset[features_]
    y = dataset[label]
    
    X = scl.fit_transform(x)
    
    model.fit(X,y)
    
    df = dataset.iloc[-1:,:].copy()
    df.edellinen = df[label]
    df[label] = np.nan
    next_day = next_wanted_weekday(pd.to_datetime(df.index.values[0]), threshold=1)
    df['maksu_pv'] = next_day
    df = df.set_index('maksu_pv')
    
    if not baseline:
    
        df['first_pay_day_distance'] = df.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
        df['second_pay_day_distance'] = df.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
        df['third_pay_day_distance'] = df.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
        df['fourth_pay_day_distance'] = df.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values
    
    
    df[label] = np.maximum(df.edellinen,model.predict(scl.transform(df[features_])))
    
    
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
        
        if not baseline:
        
            dff['first_pay_day_distance'] = dff.reset_index().apply(lambda x: first_pay_day_distance(x['maksu_pv']),axis=1).values
            dff['second_pay_day_distance'] = dff.reset_index().apply(lambda x: second_pay_day_distance(x['maksu_pv']),axis=1).values
            dff['third_pay_day_distance'] = dff.reset_index().apply(lambda x: third_pay_day_distance(x['maksu_pv']),axis=1).values
            dff['fourth_pay_day_distance'] = dff.reset_index().apply(lambda x: fourth_pay_day_distance(x['maksu_pv']),axis=1).values
            

        
        dff[label] = np.maximum(dff.edellinen,model.predict(scl.transform(dff[features_])))
        dfs.append(dff)
        current_date = next_day
    prediction = pd.concat(dfs)                   
    
    dataset['forecast'] = 'Toteutunut'
    prediction ['forecast'] ='Ennuste'
    
    result_df = pd.concat([dataset,prediction])
    
    result_df['ennusteen_pituus'] = length
    
    result_df['regularisointi'] = {'Lasso':'Lasso','Ridge':'Ridge', 'ElasticNet': 'Elastinen verkko'}[reg_type]
    
    return result_df


# Alustetaan lähtödata ja valikot.

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
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%-d . %Bta %Y'), '{:,}'.format(round(df.iloc[i][l],2)).replace(',',' ')) for i in range(len(df))]

    
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
                                                     tickfont = dict(size=14),
                                                      exponentformat= "none", 
                                                         separatethousands= True
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
                                  tickformat='%-d.%-m %Y',
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
                                                     tickfont = dict(size=14),
                                                      exponentformat= "none", 
                                                         separatethousands= True
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
                                  tickformat='%-m / %Y',
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


# Visualisoidaan haluttu muuttuja viikoittain.
def plot_weekly_data(kunta, label):
    
    df = data[data.kunta_nimi==kunta]

    l = labels[label].replace('_kum','')
    
    df = df.resample('W')[l].sum()
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(df.values[i],2)).replace(',',' ')) for i in range(len(df))]
    
    figure = go.Figure(data = [
                            go.Scatter(x = df.index,
                                       y = df.values,
                                      name = '',
                                       mode = 'lines+markers',
                                       marker = dict(color='orange'),
                                      hovertemplate = hovertemplate)
                            ],
                       layout = go.Layout()
                      )
    
    
    figure.update_layout(yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                     ),
                       title = dict(text = kunta+':<br>'+label+' viikoittain', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                       legend = dict(font=dict(size=18)),
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                       xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                                  tickformat='%-d.%-m %Y',
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
                                                     tickfont = dict(size=14),
                                                  exponentformat= "none", 
                                                         separatethousands= True
                                     ),
                       title = dict(text = kunta+':<br>'+label+' kvartaaleittain', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                        xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                                  # tickformat='%-m / %Y',
                                                
                            rangeselector=dict(
                                buttons=list([

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


# Visualisoidaan haluttu muuttuja vuosittain.
def plot_yearly_data(kunta, label):
    
    df = data[data.kunta_nimi==kunta]

    l = labels[label].replace('_kum','')
    
    df = df.resample('Y')[l].sum()
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%Y'), '{:,}'.format(round(df.values[i],2)).replace(',',' ')) for i in range(len(df))]
    
    figure = go.Figure(data = [
                            go.Scatter(x = df.index,
                                       y = df.values,
                                      name = '',
                                       mode = 'lines+markers',
                                       marker = dict(color='grey'),
                                      hovertemplate = hovertemplate)
                            ],
                       layout = go.Layout()
                      )
    
    
    figure.update_layout(yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                  exponentformat= "none", 
                                                         separatethousands= True
                                     ),
                       title = dict(text = kunta+':<br>'+label+' vuosittain', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',
                         height = 600,
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                        xaxis=dict(title = dict(text = 'Aika',
                                                font=dict(size=18, family = 'Arial Black')
                                               ),
                                   tickfont = dict(size=14),
                                #   tickformat='%-m / %Y',
                                                
                            rangeselector=dict(
                                buttons=list([

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
def plot_cum_data(df, label):
    
    kunta = df.kunta.values[0]
      
    l = labels[label]
    
    hovertemplate = ['<b>{}</b>:<br>{} €'.format(df.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(df.iloc[i][l],2)).replace(',',' ')) for i in range(len(df))]
    
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
                       title = dict(text = kunta+':<br>'+label+' kumulatiivisena', x=.5, font=dict(size=24,family = 'Arial')),
                          template = 'seaborn',   
                          height = 600,
                         hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                          xaxis = dict(title = dict(text='Aika',
                                                     font=dict(size=18, family = 'Arial Black')
                                                    ),
                                        tickfont = dict(size=14),
                                       tickformat='%-d.%-m %Y',
                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    )
                    )
    return figure    
    
# Visualisoidaan ennuste päivittäin.
def plot_daily_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    daily_baseline = df[df.forecast=='Ennuste'].daily_baseline
    
    
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(daily_true.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(daily_true.values[i],2)).replace(',',' ')) for i in range(len(daily_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(daily_pred.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(daily_pred.values[i],2)).replace(',',' ')) for i in range(len(daily_pred))]
    
    hover_baseline = ['<b>{}</b>:<br>{} €'.format(daily_baseline.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(daily_baseline.values[i],2)).replace(',',' ')) for i in range(len(daily_baseline))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = daily_true.index, 
                                   y = daily_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = daily_pred.index, 
                                   y = daily_pred.values, 
                                   name = 'Ennuste ({})'.format(reg_type), 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red')),
                            go.Scatter(x = daily_baseline.index, 
                                   y = daily_baseline.values, 
                                   name = 'Ennuste (Lineaariregressio)', 
                                   hovertemplate = hover_baseline,
                                   marker_symbol = 'diamond',
                                   mode = 'lines+markers',
                                   marker = dict(color='blue', size = 10))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       tickformat='%-d.%-m %Y',
#                                                        rangeslider=dict(visible=True),
#                                                        type='date'
                                                       
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' päivittäin (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    
    return figure


# Visualisoidaan ennuste viikoittain.
def plot_weekly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    daily_baseline = df[df.forecast=='Ennuste'].daily_baseline
    
    weekly_true = daily_true.resample('W').sum()
    weekly_pred = daily_pred.resample('W').sum()  
    weekly_baseline = daily_baseline.resample('W').sum()
    
    
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(weekly_true.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(weekly_true.values[i],2)).replace(',',' ')) for i in range(len(weekly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(weekly_pred.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(weekly_pred.values[i],2)).replace(',',' ')) for i in range(len(weekly_pred))]
    
    hover_baseline = ['<b>{}</b>:<br>{} €'.format(weekly_baseline.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(weekly_baseline.values[i],2)).replace(',',' ')) for i in range(len(weekly_baseline))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = weekly_true.index, 
                                   y = weekly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = weekly_pred.index, 
                                   y = weekly_pred.values, 
                                   name = 'Ennuste (Lineaariregressio)', 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red')),
                            go.Scatter(x = weekly_baseline.index, 
                                   y = weekly_baseline.values, 
                                   name =  'Ennuste ({})'.format(reg_type),
                                   hovertemplate = hover_baseline,
                                   marker_symbol = 'diamond',
                                   mode = 'lines+markers',
                                   marker = dict(color='blue', size = 10))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       tickformat='%-d.%-m %Y',
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' viikoittain (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    
    return figure
                       
# Visualisoidaan ennuste kvartaaleittain.                       
def plot_quaterly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    daily_baseline = df[df.forecast=='Ennuste'].daily_baseline
                       
    quaterly_true = daily_true.resample('Q').sum()
    quaterly_pred = daily_pred.resample('Q').sum()   
    quaterly_baseline = daily_baseline.resample('Q').sum()
                       
    quaterly_true.index = [to_quartals(i) for i in quaterly_true.index]
    quaterly_pred.index = [to_quartals(i) for i in quaterly_pred.index]
    quaterly_baseline.index = [to_quartals(i) for i in quaterly_baseline.index]
                       
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(quaterly_true.index[i], '{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' ')) for i in range(len(quaterly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(quaterly_pred.index[i], '{:,}'.format(round(quaterly_pred.values[i],2)).replace(',',' ')) for i in range(len(quaterly_pred))]
    
    hover_baseline = ['<b>{}</b>:<br>{} €'.format(quaterly_baseline.index[i], '{:,}'.format(round(quaterly_baseline.values[i],2)).replace(',',' ')) for i in range(len(quaterly_baseline))]
    
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = quaterly_true.index, 
                                   y = quaterly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = quaterly_pred.index, 
                                   y = quaterly_pred.values, 
                                   name = 'Ennuste ({})'.format(reg_type), 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red')),
                            go.Scatter(x = quaterly_baseline.index, 
                                   y = quaterly_baseline.values, 
                                   name = 'Ennuste (Lineaariregressio)', 
                                   hovertemplate = hover_baseline,
                                   marker_symbol = 'diamond',
                                   mode = 'lines+markers',
                                   marker = dict(color='blue', size = 10))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' kvartaaleittain (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure

# Visualisoidaan ennuste vuosittain.                       
def plot_yearly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    daily_baseline = df[df.forecast=='Ennuste'].daily_baseline
                       
    yearly_true = daily_true.resample('Y').sum()
    yearly_pred = daily_pred.resample('Y').sum()  
    yearly_baseline = daily_baseline.resample('Y').sum()  
                       
    yearly_true.index = [i.strftime('%Y')  for i in yearly_true.index]
    yearly_pred.index = [i.strftime('%Y') for i in yearly_pred.index]
    yearly_baseline.index = [i.strftime('%Y') for i in yearly_baseline.index]
                       
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(yearly_true.index[i], '{:,}'.format(round(yearly_true.values[i],2)).replace(',',' ')) for i in range(len(yearly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(yearly_pred.index[i], '{:,}'.format(round(yearly_pred.values[i],2)).replace(',',' ')) for i in range(len(yearly_pred))]
    
    hover_baseline = ['<b>{}</b>:<br>{} €'.format(yearly_baseline.index[i], '{:,}'.format(round(yearly_baseline.values[i],2)).replace(',',' ')) for i in range(len(yearly_pred))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = yearly_true.index, 
                                   y = yearly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = yearly_pred.index, 
                                   y = yearly_pred.values, 
                                   name = 'Ennuste ({})'.format(reg_type), 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red')),
                            go.Scatter(x = yearly_baseline.index, 
                                   y = yearly_baseline.values, 
                                   name = 'Ennuste (Lineaariregressio)', 
                                   hovertemplate = hover_baseline,
                                   marker_symbol = 'diamond',
                                   mode = 'lines+markers',
                                   marker = dict(color='blue', size = 10))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                         title = dict(text = kunta+':<br>'+label+' vuosittain (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure

# Visualisoidaan ennuste kuukausittain.
def plot_monthly_prediction(df):
    
    daily_true = df[df.forecast=='Toteutunut'].daily
    daily_pred = df[df.forecast=='Ennuste'].daily
    daily_baseline = df[df.forecast=='Ennuste'].daily_baseline
                       
    monthly_true = daily_true.resample('M').sum()
    monthly_pred = daily_pred.resample('M').sum()     
    monthly_baseline = daily_baseline.resample('M').sum()  
                       
    monthly_true.index = [i.strftime('%B %Y')  for i in monthly_true.index]
    monthly_pred.index = [i.strftime('%B %Y') for i in monthly_pred.index]
    monthly_baseline.index = [i.strftime('%B %Y') for i in monthly_baseline.index]
                       
    label = df.name.values[0]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(monthly_true.index[i], '{:,}'.format(round(monthly_true.values[i],2)).replace(',',' ')) for i in range(len(monthly_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(monthly_pred.index[i], '{:,}'.format(round(monthly_pred.values[i],2)).replace(',',' ')) for i in range(len(monthly_pred))]
    
    hover_baseline = ['<b>{}</b>:<brmonthly_baseline>{} €'.format(monthly_baseline.index[i], '{:,}'.format(round(monthly_baseline.values[i],2)).replace(',',' ')) for i in range(len(monthly_baseline))]
    
    figure = go.Figure(data = [
                            
                            go.Bar(x = monthly_true.index, 
                                   y = monthly_true.values, 
                                   name = 'Toteutunut',
                                   hovertemplate = hover_true,
                                   marker = dict(color='green')),
                            go.Bar(x = monthly_pred.index, 
                                   y = monthly_pred.values, 
                                   name = 'Ennuste ({})'.format(reg_type), 
                                   hovertemplate = hover_pred,
                                   marker = dict(color='red')),
                            go.Scatter(x = monthly_baseline.index, 
                                   y = monthly_baseline.values, 
                                   name = 'Ennuste (Lineaariregressio)', 
                                   hovertemplate = hover_baseline,
                                   marker_symbol = 'diamond',
                                   mode = 'lines+markers',
                                   marker = dict(color='blue', size = 10))
    
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       
                                                        ),
                                        yaxis = dict(title = dict(text = label + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          title = dict(text = kunta+':<br>'+label+' kuukausittain (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure

# Visualisoidaan ennuste kumulatiivisena.
def plot_cumulative_prediction(df):
    

    label_name = df.name.values[0]                    
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.regularisointi.values[0]
    
    df_true = df[df.forecast=='Toteutunut']
    df_pred = df[df.forecast=='Ennuste']
    
    hover_true = ['<b>{}</b>:<br>{} €'.format(df_true.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(df_true.iloc[i][label],2)).replace(',',' ')) for i in range(len(df_true))]
    
    hover_pred = ['<b>{}</b>:<br>{} €'.format(df_pred.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(df_pred.iloc[i][label],2)).replace(',',' ')) for i in range(len(df_pred))]
    
    hover_baseline = ['<b>{}</b>:<br>{} €'.format(df_pred.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(df_pred.iloc[i]['baseline'],2)).replace(',',' ')) for i in range(len(df_pred))]
    
    
    figure = go.Figure(data = [
                            
                            go.Scatter(x = df_true.index, 
                                       y = df_true[label], 
                                       name = 'Toteutunut', 
                                       hovertemplate = hover_true,
                                       marker = dict(color='green')),
                            go.Scatter(x = df_pred.index, 
                                       y = df_pred[label], 
                                       name = 'Ennuste ({})'.format(reg_type), 
                                       hovertemplate = hover_pred,
                                       marker = dict(color='red')),
                            go.Scatter(x = df_pred.index, 
                                       y = df_pred['baseline'], 
                                       name = 'Ennuste (Lineaariregressio)', 
                                       hovertemplate = hover_baseline,
                                       marker_symbol = 'diamond',
                                       marker = dict(color='blue',size=10))
    
                            ],
                       layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       tickformat='%-d.%-m %Y',
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                          legend = dict(font=dict(size=18)),
                                          height = 600,
                                          hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                          title = dict(text = kunta+':<br>'+label_name+' kumulatiivisena (ennuste)', x=.5, font=dict(size=24,family = 'Arial'))
                                          )
                      )
    
    return figure


# Visualisoidaan testitulokset päivittäin.
def plot_daily_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.reg_type.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    daily_test = test_data['ennustettu'] - test_data['ennuste_edellinen']
    daily_baseline = test_data['baseline'] - test_data['baseline_edellinen']
    daily_true = test_data[label] - test_data['edellinen']

    
    test_error = daily_test - daily_true
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / daily_true), 2)

    baseline_error = daily_baseline - daily_true
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / daily_true), 2)

    
    
    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(daily_test.index[i].strftime('%-d. %-Bta %Y'),
        '{:,}'.format(round(daily_true[i],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(daily_test[i],2)).replace(',',' '),
        '{:,}'.format(round(daily_baseline[i],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(test_data))]

        
        
    figure = go.Figure(data=[

                go.Bar(x = daily_true.index, 
                       y = daily_true.values, 
                       name = 'Testidata',
                       hovertemplate = ['{}:<br>{} €'.format(daily_true.index[i].strftime('%-d. %Bta %Y'),
                                                          '{:,}'.format(round(daily_true.values[i],2)).replace(',',' ')) for i in range(len(daily_true))],
                       marker = dict(color='green')),
                go.Scatter(x = daily_test.index, 
                           y = daily_test.values, 
                           name = 'Ennuste ({})'.format(reg_type),
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red', size = 10)),
                go.Scatter(x = daily_baseline.index, 
                           y = daily_baseline.values, 
                           name = 'Ennuste (Lineaariregressio)',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker_symbol = 'diamond',
                           marker = dict(color='blue', size = 10)),
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       tickformat = '%-d.%-m %Y',
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' päivittäin (testi)',x=.5, font=dict(size=24,family = 'Arial')))
                      )
    
    return figure


# Visualisoidaan testitulokset viikoittain.
def plot_weekly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.reg_type.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    daily_test = test_data['ennustettu'] - test_data['ennuste_edellinen']
    daily_baseline = test_data['baseline'] - test_data['baseline_edellinen']
    daily_true = test_data[label] - test_data['edellinen']
    
    weekly_test = daily_test.resample('W').sum()
    weekly_true = daily_true.resample('W').sum()
    weekly_baseline = daily_baseline.resample('W').sum()

    
    test_error = weekly_test - weekly_true
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / weekly_true), 2)
    
    baseline_error = weekly_baseline - weekly_true
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / weekly_true), 2)


    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(weekly_test.index[i].strftime('%-d. %-Bta %Y'),
        '{:,}'.format(round(weekly_true[i],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(weekly_test[i],2)).replace(',',' '),
        '{:,}'.format(round(weekly_baseline[i],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(weekly_test))]

        
        
    figure = go.Figure(data=[

                go.Bar(x = weekly_true.index, 
                       y = weekly_true.values, 
                       name = 'Testidata',
                       hovertemplate = ['{}:<br>{} €'.format(weekly_true.index[i].strftime('%-d. %Bta %Y'),
                                                          '{:,}'.format(round(weekly_true.values[i],2)).replace(',',' ')) for i in range(len(weekly_true))],
                       marker = dict(color='green')),
                go.Scatter(x = weekly_test.index, 
                           y = weekly_test.values, 
                           name = 'Ennuste ({})'.format(reg_type),
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red', size = 10)),
                go.Scatter(x = weekly_baseline.index, 
                           y = weekly_baseline.values, 
                           name = 'Ennuste (Lineaariregressio)',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker_symbol = 'diamond',
                           marker = dict(color='blue', size = 10))
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                       tickformat = '%-d.%-m %Y',
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),  
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' viikoittain (testi)',x=.5, font=dict(size=24,family = 'Arial')))
                      )
    
    return figure
    
# Visualisoidaan testitulokset kuukausittain.    
def plot_monthly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.reg_type.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    daily_test = test_data['ennustettu'] - test_data['ennuste_edellinen']
    daily_baseline = test_data['baseline'] - test_data['baseline_edellinen']
    daily_true = test_data[label] - test_data['edellinen']
    
    
    
    
    monthly_test = daily_test.resample('M').sum()
    monthly_true = daily_true.resample('M').sum()
    monthly_baseline = daily_baseline.resample('M').sum()
          
    test_error = monthly_true - monthly_test
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / monthly_true), 2)

    baseline_error = monthly_true - monthly_baseline
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / monthly_true), 2)


    monthly_true.index = [i.strftime('%B %Y')  for i in monthly_true.index]
    monthly_test.index = [i.strftime('%B %Y') for i in monthly_test.index]
    monthly_baseline.index = [i.strftime('%B %Y') for i in monthly_baseline.index]
    
    hover_true = ['{}:<br>{} €'.format(monthly_true.index[i],
                                                          '{:,}'.format(round(monthly_true.values[i],2)).replace(',',' ')) for i in range(len(monthly_true))]
    

    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(monthly_test.index[i],
        '{:,}'.format(round(monthly_true[i],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(monthly_test[i],2)).replace(',',' '),
        '{:,}'.format(round(monthly_baseline[i],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(monthly_test))]
    
    
    
    figure = go.Figure(data=[

                go.Bar(x = monthly_true.index, 
                       y = monthly_true.values, 
                       name = 'Testidata', 
                       hovertemplate = hover_true, 
                       marker = dict(color='green')),
                go.Scatter(x = monthly_test.index,
                           y = monthly_test.values, 
                           name = 'Ennuste ({})'.format(reg_type),
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red', size = 10)),
        
                go.Scatter(x = monthly_baseline.index, 
                           y = monthly_baseline.values, 
                           name = 'Ennuste (Lineaariregressio)',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker_symbol = 'diamond',
                           marker = dict(color='blue', size = 10))
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kuukausittain (testi)',x=.5, font=dict(size=24,family = 'Arial'))
                                       )
                      ) 
    return figure


# Visualisoidaan testitulokset vuosittain.    
def plot_yearly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.reg_type.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    daily_test = test_data['ennustettu'] - test_data['ennuste_edellinen']
    daily_baseline = test_data['baseline'] - test_data['baseline_edellinen']
    daily_true = test_data[label] - test_data['edellinen']
       
    
    yearly_test = daily_test.resample('Y').sum()
    yearly_true = daily_true.resample('Y').sum()
    yearly_baseline = daily_baseline.resample('Y').sum()   
         
    test_error = yearly_true - yearly_test
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / yearly_true), 2)
    
    baseline_error = yearly_true - yearly_baseline
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / yearly_true), 2)


    yearly_true.index = [i.strftime('%Y')  for i in yearly_true.index]
    yearly_test.index = [i.strftime('%Y') for i in yearly_test.index]
    yearly_baseline.index = [i.strftime('%Y') for i in yearly_baseline.index]
    
    hover_true = ['{}:<br>{} €'.format(yearly_true.index[i],
                                                          '{:,}'.format(round(yearly_true.values[i],2)).replace(',',' ')) for i in range(len(yearly_true))]

    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(yearly_test.index[i],
        '{:,}'.format(round(yearly_true[i],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(yearly_test[i],2)).replace(',',' '),
        '{:,}'.format(round(yearly_baseline[i],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(yearly_test))]
    
    
    
    figure = go.Figure(data=[

                go.Bar(x = yearly_true.index, 
                       y = yearly_true.values, 
                       name = 'Testidata', 
                       hovertemplate = hover_true, 
                       marker = dict(color='green')),
                go.Scatter(x = yearly_test.index,
                           y = yearly_test.values, 
                           name = 'Ennuste ({})'.format(reg_type),
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red', size = 10)),
                go.Scatter(x = yearly_baseline.index, 
                           y = yearly_baseline.values, 
                           name = 'Ennuste (Lineaariregressio)',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker_symbol = 'diamond',
                           marker = dict(color='blue', size = 10))
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' vuosittain (testi)',x=.5, font=dict(size=24,family = 'Arial'))
                                       )
                      ) 
    return figure
    
# Visualisoidaan testitulokset kvartaaleittain.
def plot_quaterly_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    kunta = df.kunta.values[0]
    reg_type = df.reg_type.values[0]
    
    train_data = df[df.split=='train']
    val_data = df[df.split=='val']
    test_data = df[df.split=='test']
    
    daily_train = train_data[label] - train_data['edellinen']
    daily_val = val_data[label] - val_data['edellinen']
    daily_test = test_data['ennustettu'] - test_data['ennuste_edellinen']
    daily_baseline = test_data['baseline'] - test_data['baseline_edellinen']
    daily_true = test_data[label] - test_data['edellinen']
    
    quaterly_test = daily_test.resample('Q').sum()
    quaterly_true = daily_true.resample('Q').sum()
    quaterly_baseline = daily_baseline.resample('Q').sum()

    
    test_error = quaterly_true - quaterly_test
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / quaterly_true), 2)
    
    baseline_error = quaterly_true - quaterly_baseline
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / quaterly_true), 2)

    
    
    quaterly_true.index = [to_quartals(i) for i in quaterly_true.index]
    quaterly_test.index = [to_quartals(i) for i in quaterly_test.index]
    quaterly_baseline.index = [to_quartals(i) for i in quaterly_baseline.index]
    
    
    hover_true = ['{}:<br>{} €'.format(quaterly_true.index[i],'{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' ')) for i in range(len(quaterly_true))]
    
    hover_test = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennustettu</b>: {} €<br><b>Virhe</b>: {} €<br><b>Tarkkuus</b>: {} %'.format(quaterly_true.index[i],'{:,}'.format(round(quaterly_true.values[i],2)).replace(',',' '),'{:,}'.format(round(quaterly_test.values[i],2)).replace(',',' '),'{:,}'.format(round(test_error.values[i],2)).replace(',',' '),round(test_error_percentage.values[i],2)) for i in range(len(quaterly_true))]
    
    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(quaterly_test.index[i],
        '{:,}'.format(round(quaterly_true[i],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(quaterly_test[i],2)).replace(',',' '),
        '{:,}'.format(round(quaterly_baseline[i],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(quaterly_test))]
        
    figure = go.Figure(data=[

                go.Bar(x = quaterly_true.index, 
                       y = quaterly_true.values, 
                       name = 'Testidata',
                       hovertemplate=hover_true, 
                       marker = dict(color='green')),
                go.Scatter(x = quaterly_test.index, 
                           y = quaterly_test.values, 
                           name = 'Ennuste ({})'.format(reg_type),
                           mode = 'markers', 
                           hovertemplate=hovertemplate,
                           marker = dict(color='red', size = 10)),
                go.Scatter(x = quaterly_baseline.index, 
                           y = quaterly_baseline.values, 
                           name = 'Ennuste (Lineaariregressio)',
                           mode = 'markers', 
                           hovertemplate=hovertemplate, 
                           marker_symbol = 'diamond',
                           marker = dict(color='blue', size = 10))
           
           

        ],
                       layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14)
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                        legend = dict(font=dict(size=18)),
                                        height = 600,
                                        hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kvartaaleittain (testi)',x=.5, font=dict(size=24,family = 'Arial'))
                                       )
                      )
    return figure
    
# Visualisoidaan testitulokset kumulatiivisena.    
def plot_cumulative_test(df):
    
    label_name = df.name.values[0]
    label = labels[label_name]
    reg_type = df.reg_type.values[0]
    
    kunta = df.kunta.values[0]
    
    test_data = df[df.split=='test']
    val_data = df[df.split=='val']
    train_data = df[df.split=='train']

    test_error = test_data.ennustettu - test_data[label]
    test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / test_data[label]), 2)
    
    baseline_error = test_data.baseline - test_data[label]
    baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / test_data[label]), 2)

    
    hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(test_data.index[i].strftime('%-d. %Bta %Y'),
        '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(test_data.iloc[i]['ennustettu'],2)).replace(',',' '),
        '{:,}'.format(round(test_data.iloc[i]['baseline'],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(test_data))]
    

    
    
    figure = go.Figure(data=[

                go.Scatter(x = train_data.index, 
                           y = train_data[label], 
                           name = 'Opetusdata', 
                           hovertemplate = ['{}:<br>{} €'.format(train_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(train_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(train_data))],
                           marker = dict(color = 'purple')),
                go.Scatter(x = val_data.index, 
                           y = val_data[label], 
                           name = 'Validointidata', 
                           hovertemplate = ['{}:<br>{} €'.format(val_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(val_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(val_data))],
                           marker = dict(color = 'orange')),

                go.Scatter(x = test_data.index, 
                           y = test_data[label], 
                           name = 'Testidata', 
                           hovertemplate = ['{}:<br>{} €'.format(test_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(test_data))],
                           marker = dict(color='green')),
                go.Scatter(x = test_data.index, 
                           y = test_data.ennustettu, 
                           name = 'Ennuste ({})'.format(reg_type), 
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red')),
                go.Scatter(x = test_data.index, 
                           y = test_data['baseline'], 
                           name = 'Ennuste (Lineaariregressio)', 
                           hovertemplate = hovertemplate,
                           marker = dict(color='blue')),

            ],
                           layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                   ),
                                                       tickfont = dict(size=14),
                                                         tickformat='%-d.%-m %Y',
                                                        ),
                                        yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                  font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14),
                                                     
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                    ),
                                            legend = dict(font=dict(size=18)),
                                            height = 600,
                                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kumulatiivisena (testi)',x=.5, font=dict(size=24,family = 'Arial'))
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

                    ],xs =12, sm=12, md=5, lg=4, xl=4, align = 'center'),
                
                  
                   
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
                   ],xs =12, sm=12, md=5, lg=8, xl=8)
                ],justify = 'center', style = {'margin' : '10px 10px 10px 10px'}),
        

        dbc.Row(children = [
        
            dbc.Col(id = 'original_graph_col',xs =12, sm=12, md=12, lg=5, xl=5, align = 'start'),
            

            dbc.Col([dbc.RadioItems(id = 'orig_resampler', 
                                      options = [{'label':'Päivittäin','value':'D'}, 
                                                 {'label':'Viikoittain','value':'W'},
                                                 {'label':'Kuukausittain', 'value':'M'}, 
                                                 {'label':'Kvartaaleittain', 'value': 'Q',},
                                                 {'label':'Vuosittain','value':'Y'}
                                          
                                                ],
                                      className="form-group",
                                      inputClassName="form-check",
                                      labelClassName="btn btn-outline-warning",
                                      labelCheckedClassName="active",
                                      
                                      value = 'D',
                                     labelStyle={'font-size':22, 'display':'block'}
                                     ),
                    html.Br(),
                                ],
                                style = {'textAlign':'center'},
                               xs =12, sm=12, md=12, lg=2, xl=2, align = 'center'
                               ),

            
              dbc.Col(id = 'cumulative_graph_col',xs =12, sm=12, md=12, lg=5, xl=5, align = 'start')
        
        
        
        ],justify = 'center', style = {'margin' : '10px 10px 10px 10px'}),
        
        html.Br(),
        
        
        dbc.Row(children = [
        
            dbc.Col(children = [
            
                       html.H3('Valitse testi -ja validointidatan pituus.',style={'textAlign':'center', 'color':'black'}),
                       html.Br(),
                       dcc.Slider(id = 'test_slider',
                                 min = 10,
                                 max = 365,
                                 value = 100,
                                 step = 1,
                                 marks = {10: {'label':'10 päivää', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                        #  30:{'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                        #  70: {'label':'70 päivää', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          100:{'label':'100 päivää', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                         # 150:{'label':'150 päivää', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          180:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                          365:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}}

                                          }
                                 ),
                      html.Br(),  
                      html.Div(id = 'test_size_indicator') 

            
            ],xs =12, sm=12, md=12, lg=6, xl=6, align = 'start'),
            
            dbc.Col([html.H3('Valitse regularisointityyppi.',style={'textAlign':'center', 'color':'black'}),
                     html.Br(),
                    dcc.RadioItems(id = 'reg_type',
                                   options=[
                                       {'label': 'Ridge', 'value': 'Ridge'},
                                       {'label': 'Lasso', 'value': 'Lasso'},
                                       {'label': 'Elastinen verkko', 'value': 'ElasticNet'}
                                   ],
                                   value='Lasso',
                                   labelStyle={'display':'inline-block', 'padding':'10px'},
                                   
                           
                                   style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}
                                ),
                    ],xs =12, sm=12, md=12, lg=6, xl=6, align = 'start'),

        
        
        ], justify = 'center', style = {'margin' : '10px 10px 10px 10px'}),


       
        dbc.Row([ dbc.Col(xs =5, sm=5, md=5, lg=5, xl=5, align = 'start'),
                  dbc.Col([

                      dbc.Button('Testaa',
                                  id='start_button',
                                  n_clicks=0,
                                  outline=False,
                                  className="btn btn-outline-info",
                                  style = dict(fontSize=36)
                                  )
                  ],
                          xs =2, sm=2, md=2, lg=2, xl=2, align = 'center'),
                  dbc.Col(xs =5, sm=5, md=5, lg=5, xl=5, align = 'end')  

                        
                    
                     
                    ],justify='center', style = {'margin' : '10px 10px 10px 10px'}),
        
        html.Br(),
        
        dbc.Tabs(children= [
            
            dbc.Tab(label = 'Tulokset',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28}, 
                    children = [
       
                        html.Br(),

                        dcc.Loading(
                            html.Div(id = 'results', style = {'margin' : '10px 10px 10px 10px'}),
                                   type = spinners[random.randint(0,len(spinners)-1)]),
                        html.Div(id = 'hidden_data_div',
                                 children= [
                                            dcc.Store(id='predict_store'),
                                            dcc.Store(id='test_store'),
                                            dcc.Store(id='intermediate_store'),
                                            dcc.Store(id = 'cum_data_store'),
                                            dcc.Download(id = "download-component")
                                           ]
                                 )
                    ]
                    

                                   ),
            dbc.Tab(label = 'Ohje ja esittely',
                   tabClassName="flex-grow-1 text-center",                    
                   tab_style = {'font-size':28},
                   children = [ 
                       dbc.Row(justify='center', children=[
                       
                           dbc.Col(xs =12, sm=12, md=5, lg=6, xl=6, children =[
                   

                                html.Br(),
                                html.H4('Johdanto',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P('Tämä sovellus hyödyntää Kelan tilastoimia päivittäistä dataa toimeentulotukien maksuista ja palautuksista ja pyrkii muodostamaan koneoppimisen avulla ennusteen tulevaisuuteen. Käyttäjä voi valita haluamansa tukilajin sekä kunnan alasvetovalikoista, tarkastella toteumadataa sekä tehdä valitun pituisen ennusteen. ',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('Sovelluksen avulla voidaan testata eri lineaarisen regression varianttien toimivuutta toimeentulotuen maksusuoritusten ennustamiseen. Ennustaminen tapahtuu muodostamalla halutun suureen kumulatiivinen aikasarja, joka useimmiten paljastuu lineaariseksi. Ennustamalla kumulatiivista arvoa, saadaan päiväkohtainen arvo vähentämällä saadusta kumulatiivisesta arvosta edellisen päivän kumulatiivinen arvo. Päiväkohtaiset ennusteet voidaan taas summata eri ajanjaksoittain, esimerkiksi kvartaaleittain. Ennusteen piirteinä hyödynnetään edeltävän päivän kumulatiivista arvoa sekä ajallista etäisyyttä nykyhetken ja tulevan maksupäivän välillä (päivissä). Tarkemmin sanoen, käytössä on etäisyyden käänteisluku, jolla pyritään mallintamaan läheisyyttä maksupäivään suuremmilla luvuilla. Jos esimerkiksi seuraava maksupäivä on kolmen päivän päästä, on käänteinen etäisyys 1/3. Etäisyyden ollessa nolla (on maksupäivä), käänteinen etäisyys on suurin ja saa arvo yksi. Lisäksi käänteinen etäisyys on normalisoitu kertomalla yhdestä poikkeavat luvut kertoimella {}, jotta saadaan tehtyä ero käänteiselle etäisyydelle, kun etäisyys on yksi tai nolla.'.format(distance_baseline),style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('Sovellus hyödyntää käyttäjän valitsemaa regularisointimallia. Valittavissa on Ridge, Lasso sekä niiden yhdistelmä, Elastinen verkko. Ohjelma optimoi algoritmin regularisointiparametrin ja suorittaa lopullisen ennusteen parhaalla mahdollisella algoritmilla. Regularisoinnista voi lukea lisää alla esitettyjen lähdeviittauksien kautta. Ohjelmassa on valittu viitearvoksi tavallinen lineaariregressio, joka muodostaa suoran vain edellisten päivien kumulatiivisten arvojen perusteella. Näin voidaan tarkastella pystyttiinkö monimutkaisemmalla mallinnuksella tuottamaan yksinkertaista mallia parempi ennuste.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('Ennusteen laatua pystyy tarkastelemaan vertailemalla toteutunutta dataa sekä testissä tehtyä ennustetta. Näin käyttäjä saa parempaa tietoa ennusteen luotettavuudesta. Tässä sovelluksessa data esitetään kuvaajilla, joiden aikafrekvenssiä voi säätää haluamakseen. Kuvaajien oikeassa yläkulmassa on työkaluja muun muassa kuvan tarkentamiseen sekä kuvatiedoston vientiin.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.H4('Ohje',style={'textAlign':'center', 'color':'black'}),
                                html.Br(),
                                html.P('1. Valitse haluttu kunta.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('2. Valitse haluttu suure. Voit tarkastella suuretta eri ajanjaksoissa tai kumulatiivisesti.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),             
                                html.Br(),
                                html.P('3. Valitse validointi - ja testidatan pituus. Tästä puolet käytetään validointidatana, jonka avulla koneoppimisalgoritmi valitsee hyperparametrit. Toinen puoli käytetään testaamiseen.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('4. Valitse regularisointityypiksi joko Ridge, Lasso tai Elastinen verkko.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('5. Klikkaa "Testaa"-painiketta. Tämän jälkeen voit tarkastella testin tuloksia alle ilmestyvästä kuvaajasta.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('6. Analysoi testin tuloksia. Tuloksia voi tarkastella kuvaajalla eri aikayksiköittäin. Voit myös halutessasi toistaa aiemmat toimenpiteet eri lähtömuuttujilla.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('7. Valitse ennusteen pituus. Lopullinen algoritmi laskee ennusteen halutulle ajalle.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('8. Klikkaa ennusta. Alle ilmestyy kuvaaja, jossa on ennusteet valitulle ajalle.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('9. Tarkastele ennustetuloksia. Tulokset saa kuvaajaan valitun aikatiheyden mukaan.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                html.Br(),
                                html.P('10. Halutessasi voit viedä tulostiedoston Exceliin klikkaamalla "Lataa tiedosto koneelle" -painiketta.',style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
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
                                           html.Label(['Kela: ', 
                                                    html.A('Kelan maksaman perustoimeentulotuen menot ja palautukset', href = "https://www.avoindata.fi/data/fi/dataset/kelan-maksaman-perustoimeentulotuen-menot-ja-palautukset",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.Br(),
                                           html.Label(['Wikipedia: ', 
                                                    html.A('Lasso-regressio (englanniksi)', href = "https://en.wikipedia.org/wiki/Lasso_regression",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.Label(['Wikipedia: ', 
                                                    html.A('Ridge-regressio (i.e. Tikhonov regularisaatio, englanniksi)', href = "https://en.wikipedia.org/wiki/Tikhonov_regularization",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'}),
                                           html.Br(),
                                           html.Label(['Wikipedia: ', 
                                                    html.A('Elastinen verkko -regressio (englanniksi)', href = "https://en.wikipedia.org/wiki/Elastic_net_regularization",target="_blank")
                                                   ],style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})

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
                                                    html.A('GitHub:ssa', href='https://github.com/tuopouk/toimeentulotuki', 
                                                           target = '_blank'
                                                          )
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
    
    return [html.P('Valitsit {} päivää validointi -ja testidataksi.'.format(value),style = {'textAlign':'center', 'fontSize':20, 'fontFamily':'Arial Black', 'color':'black'})]

@app.callback(
    Output('forecast_slider_indicator','children'),
    [Input('forecast_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.P('Valitsit {} päivän ennusteen.'.format(value),style = {'textAlign':'center', 'fontSize':20, 'fontFamily':'Arial Black', 'color':'black'})]


@app.callback(
    
    
    ServersideOutput('cum_data_store','data'),
    [Input('kunta_dropdown','value')]
)
def store_cum_data(kunta):
    
    df = get_cum_city_data(kunta)
    df['kunta'] = kunta
    
    return df
    

@app.callback(
    [Output('results','children'),
    ServersideOutput('test_store','data'),
    ServersideOutput('intermediate_store','data')],
    [Input('start_button', 'n_clicks'),
     State('cum_data_store','data'),
    State('label_dropdown', 'value'),
     State('reg_type', 'value'),
    State('test_slider', 'value')
    ]
)
def start(n_clicks, cum_data, label_name, reg_type, test):
    
    if n_clicks > 0:
    
        label = labels[label_name]
        kunta = cum_data.kunta.values[0]

        dataset = shift_dates(dataset = cum_data, label = label)

        train_val_test_df = train_val_test(dataset, label, reg_type, test)
        
        train_val_test_df['daily_true'] = train_val_test_df[label] - train_val_test_df['edellinen']
        train_val_test_df['daily_pred'] = train_val_test_df['ennustettu'] - train_val_test_df['ennuste_edellinen']
        train_val_test_df['daily_baseline'] = train_val_test_df['baseline'] - train_val_test_df['baseline_edellinen']
        train_val_test_df['name'] = label_name
        train_val_test_df['kunta'] = kunta

        alpha = train_val_test_df.alpha.dropna().values[0]

        test_data = train_val_test_df[train_val_test_df.split=='test']
        val_data = train_val_test_df[train_val_test_df.split=='val']
        train_data = train_val_test_df[train_val_test_df.split=='train']

        test_error = test_data.ennustettu - test_data[label]
        test_error_percentage = np.round( 100 * (1 - np.absolute(test_error) / test_data[label]), 2)
        
        baseline_error = test_data.baseline - test_data[label]
        baseline_error_percentage = np.round( 100 * (1 - np.absolute(baseline_error) / test_data[label]), 2)

        hovertemplate = ['<b>{}</b><br><b>Toteutunut</b>: {} €<br><b>Ennuste ({})</b>: {} €<br><b>Ennuste (Lineaariregressio)</b>: {} €<br><b>Ennustevirhe</b>: {} €<br><b>Ennustetarkkuus</b>: {} %<br><b>LR virhe</b>: {} €<br><b>LR tarkkuus</b>: {} %.'.format(test_data.index[i].strftime('%-d. %Bta %Y'),
        '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' '),
         reg_type,
        '{:,}'.format(round(test_data.iloc[i]['ennustettu'],2)).replace(',',' '),
        '{:,}'.format(round(test_data.iloc[i]['baseline'],2)).replace(',',' '),
        '{:,}'.format(round(test_error[i],2)).replace(',',' '),
        round(test_error_percentage[i],2),
        '{:,}'.format(round(baseline_error[i],2)).replace(',',' '),
        round(baseline_error_percentage[i],2)
        ) for i in range(len(test_data))]

        train_val_test_fig = go.Figure(data=[

                go.Scatter(x = train_data.index, 
                           y = train_data[label], 
                           name = 'Opetusdata',
                           hovertemplate = ['{}:<br>{} €'.format(train_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(train_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(train_data))],
                           marker = dict(color = 'purple')),
                go.Scatter(x = val_data.index, 
                           y = val_data[label], 
                           name = 'Validointidata', 
                           hovertemplate = ['{}:<br>{} €'.format(val_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(val_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(val_data))],
                           marker = dict(color = 'orange')),

                go.Scatter(x = test_data.index, 
                           y = test_data[label], 
                           name = 'Testidata', 
                           hovertemplate = ['{}:<br>{} €'.format(test_data.index[i].strftime('%-d. %Bta %Y'),
                                                             '{:,}'.format(round(test_data.iloc[i][label],2)).replace(',',' ')) for i in range(len(test_data))],
                           marker = dict(color='green')),
            
                go.Scatter(x = test_data.index, 
                           y = test_data['baseline'], 
                           name = 'Ennuste (Lineaariregressio)', 
                           hovertemplate = hovertemplate,
                           marker = dict(color='blue')),
                 
                go.Scatter(x = test_data.index, 
                           y = test_data.ennustettu, 
                           name = 'Ennuste ({})'.format({'Lasso':'Lasso','Ridge':'Ridge', 'ElasticNet': 'Elastinen verkko'}[reg_type]),
                           hovertemplate=hovertemplate, 
                           marker = dict(color='red')),

            ],
                           layout=go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                      font=dict(size=18, family = 'Arial Black')
                                                                     ),
                                                         tickfont = dict(size=14),
                                                         tickformat='%-d.%-m %Y'
                                                         
                                                        ),
                                        yaxis = dict(title = dict(text=label_name + ' (€)',
                                                                 font=dict(size=16, family = 'Arial Black')
                                                                 ),
                                                     tickfont = dict(size=14),
                                                     exponentformat= "none", 
                                                         separatethousands= True
                                                      ),
                                            legend = dict(font=dict(size=18)),
                                            height = 600,
                                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                        title = dict(text=kunta+':<br>'+label_name+' kumulatiivisena (testi)',x=.5, font=dict(size=24,family = 'Arial'))))
        
        
        frequency_options = [{'label':'Päivittäin','value':'D'},
                             {'label':'Viikoittain','value':'W'},
                             {'label':'Kuukausittain', 'value':'M'}]

        frequency_options.append({'label':'Kumulatiivisena', 'value': 'KUM'})
        
        return [dbc.Row(children = [
                    dbc.Col(xs =12, sm=12, md=12, lg=3, xl=3, align = 'start'),
                    dbc.Col(children = [
                       html.Br(),
                       html.H3('Valitse ennusteen pituus.',style={'textAlign':'center', 'color':'black'}),
                       dcc.Slider(id = 'forecast_slider',
                                 min = 30,
                                 max = 2*365,
                                 value = 90,
                                 step = 1,
                                 marks = {30: {'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                         # 90: {'label':'kolme kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                        #  120: {'label':'neljä kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                  #       180:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                         365:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                         2*365:{'label':' kaksi vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}},
                                       #  4*365:{'label':'neljä vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'black'}}

                                          }
                                 ),
                       html.Br(),
                       html.Div(id = 'forecast_slider_indicator', 
                                children = [html.P('Valitsit {} päivän ennusteen.'.format(90),style = {'textAlign':'center', 'fontSize':24, 'fontFamily':'Arial Black', 'color':'black'})])
            
            
            
            
                        ],xs =12, sm=12, md=12, lg=6, xl=6, align = 'start'),
                    dbc.Col(xs =12, sm=12, md=12, lg=3, xl=3, align = 'start'),
            
        ],justify = 'center', style = {'margin' : '10px 10px 10px 10px'}),
                
            html.Br(),    
            dbc.Row([
                    dbc.Col(xs =5, sm=5, md=5, lg=5, xl=5, align = 'start'),
                    dbc.Col(children = [dbc.Button('Ennusta',
                                  id='predict_button',
                                  n_clicks=0,
                                  outline=False,
                                  className="btn btn-outline-info",
                                  style = dict(fontSize=36)
                                          )
                                   ],xs =2, sm=2, md=2, lg=2, xl=2, align = 'center'
                         ),
                    dbc.Col(xs =5, sm=5, md=5, lg=5, xl=5, align = 'end')
            
                 ],justify = 'center', style = {'margin' : '10px 10px 10px 10px'}
             ),
                
            html.Br(),

            dbc.Row(children=[
                    dbc.Col([

                             dbc.Card(dcc.Graph(id = 'train_val_test_fig', 
                                                config=config_plots,
                                                figure =train_val_test_fig),
                                      color='dark',body=True),
                             html.P('λ = '+str(alpha)),
                             html.Br(),
                             html.P('Tällä kuvaajalla voit tarkastella ennustettua muuttujaa oikean aikayksikön valinnan mukaan. Kuvaajassa vihreällä värillä on esitetty testidata sekä punaisella testissä tehty ennuste. Sinisellä värillä on kuvattu tavallisen lineaariregression tulos. Tooltipissä on kuvattu ennustetarkkuus ja virhe niin ennusteelle kuin lineaariselle regressiollekin. Kumulatiivisessa kuvaajassa on piirretty myös opetus -ja validointidata, joita on hyödynnety regressiomallin opetuksessa sekä algoritmin optimoinnissa. Kuvaaja näyttää siis testin ja toteutuneen datan välisen eron. Yleisesti ottaen virhe on suurin päivittäisissä ennusteissa ja pienin kumulatiivisessa ennusteessa. Tulosexceliin saa koosteen lasketuista virheistä. Tässä kuvaajassa ennusteen ja toteutuneen eroa voi tarkastella paremmin aikayksiköittäin. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.', style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
                        ],xs =12, sm=12, md=12, lg=5, xl=5, align = 'start'
                    ),
                
                   dbc.Col(id = 'frequency_placeholder', children =
                           [dbc.RadioItems(id = 'resampler', 
                                           options = frequency_options,
                                      className="form-group",
                                      inputClassName="form-check",
                                      labelClassName="btn btn-outline-warning",
                                      labelCheckedClassName="active",
                                      
                                      value = None,
                                     labelStyle={'font-size':22, 'display':'block'}
                                     ),
                            html.Br()
                                ],
                                style = {'textAlign':'center'},
                               xs =12, sm=12, md=12, lg=2, xl=2, align = 'start'
                               ),

                   dbc.Col(id = 'predict_placeholder',

                           xs =12, sm=12, md=12, lg=5, xl=5, align = 'start')
               ],justify='center', style = {'margin' : '10px 10px 10px 10px'}),
                
                html.Br(),
                
                dbc.Row(children=[
                    
                    dbc.Col(xs =4, sm=4, md=5, lg=4, xl=4, align = 'start'),
                    dbc.Col(id = 'download_button_placeholder', 

                            xs =3, sm=3, md=3, lg=3, xl=3, align = 'center'),
                    dbc.Col(xs =4, sm=4, md=4, lg=4, xl=4, align = 'end')
                        ],justify='center', style = {'margin' : '10px 10px 10px 10px'})
               ], train_val_test_df, dataset
                
    

@app.callback(

    [ServersideOutput('predict_store', 'data'),
    Output('predict_placeholder', 'children'),
    Output('resampler', 'options'),
    Output('download_button_placeholder', 'children')],
    [Input('predict_button','n_clicks'),
    State('test_store', 'data'),
    State('intermediate_store', 'data'),
    State('forecast_slider','value')]

)
def predict_with_test_results(n_clicks, train_val_test, dataset, length):
    
    if n_clicks > 0:
    
       

        alpha = train_val_test.alpha.values[0]
        reg_name = train_val_test.reg_type.values[0]
        label = train_val_test.label.values[0]
        kunta = train_val_test.kunta.values[0]  
        label_name = train_val_test.name.values[0]

        reg_type = {'Lasso':'Lasso','Ridge':'Ridge','Elastinen verkko':'ElasticNet'}[reg_name]



        prediction = predict(dataset, label, length, alpha, reg_type)       


        baseline = predict(dataset, label, length, alpha, reg_type, baseline = True)

        prediction['daily'] = prediction[label] - prediction['edellinen']
        prediction['name'] = label_name
        prediction['kunta'] = kunta


        baseline['daily'] = baseline[label] - baseline['edellinen']

        prediction['baseline'] = baseline[label]
        prediction['daily_baseline'] = baseline['daily']


        true_df = prediction[prediction.forecast=='Toteutunut']
        pred_df = prediction[prediction.forecast=='Ennuste']


        hover_true = ['<b>{}</b>:<br>{} €'.format(true_df.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(true_df.iloc[i][label],2)).replace(',',' ')) for i in range(len(true_df))]

        hover_pred = ['<b>{}</b>:<br>{} €'.format(pred_df.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(pred_df.iloc[i][label],2)).replace(',',' ')) for i in range(len(pred_df))]

        hover_baseline = ['<b>{}</b>:<br>{} €'.format(pred_df.index[i].strftime('%-d. %Bta %Y'), '{:,}'.format(round(pred_df.iloc[i]['baseline'],2)).replace(',',' ')) for i in range(len(pred_df))]



        predict_fig = go.Figure(data =[

                go.Scatter(x = true_df.index, 
                           y = true_df[label], 
                           name = 'Toteutunut', 
                           hovertemplate = hover_true,
                           marker = dict(color='green')),
                go.Scatter(x = pred_df.index, 
                           y = pred_df[label], 
                           name = 'Ennuste ({})'.format({'Lasso':'Lasso','Ridge':'Ridge', 'ElasticNet': 'Elastinen verkko'}[reg_type]), 
                           hovertemplate = hover_pred,
                           marker = dict(color='red')),
                go.Scatter(x = pred_df.index, 
                           y = pred_df['baseline'], 
                           name = 'Ennuste (Lineaariregressio)', 
                           hovertemplate = hover_baseline,
                           marker = dict(color='blue'))

            ],
                           layout = go.Layout(xaxis = dict(title = dict(text='Aika',
                                                                          font=dict(size=18, family = 'Arial Black')
                                                                       ),
                                                           tickfont = dict(size=14),
                                                          tickformat='%-d.%-m %Y'
                                                          ), 
                                              yaxis = dict(title = dict(text = label_name + ' (€)',
                                                                       font=dict(size=16, family = 'Arial Black')
                                                                       ),
                                                           tickfont = dict(size=14),
                                                          exponentformat= "none", 
                                                         separatethousands= True
                                                          ),
                                              legend = dict(font=dict(size=18)),
                                              height = 600,
                                              hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                             title = dict(text = kunta+':<br>'+label_name+' kumulatiivisena (ennuste)', 
                                                          x=.5, font=dict(size=24,family = 'Arial'))
                                              )
                                   )
        frequency_options = [{'label':'Päivittäin','value':'D'},
                                 {'label':'Viikoittain','value':'W'},
                                 {'label':'Kuukausittain', 'value':'M'}]

        if length >= 90:
            frequency_options.append({'label':'Kvartaaleittain', 'value': 'Q'})
        if length >= 365:
            frequency_options.append({'label':'Vuosittain', 'value': 'Y'})

        frequency_options.append({'label':'Kumulatiivisena', 'value': 'KUM'})



        button_children = [
                               dbc.Button(children=[
                                   html.I(className="fa fa-download mr-1"), ' Lataa tiedosto koneelle'],
                                       id='download_button',
                                       n_clicks=0,
                                       outline=True,
                                       size = 'lg',
                                       color = 'light',
                                       style={'color':'black','font-family':'Arial','font-size':20}
                                       )
                                ]

        predict_placeholder_children = [dbc.Card(dcc.Graph(id = 'predict_fig', 
                                                           config=config_plots,
                                                   figure =predict_fig),
                                         color='dark',
                                         body=True),
                               html.Br(),
                               html.P('Tässä kuvaajassa esitetään itse ennuste, jota voi tarkastella, testitulosten tavoin halutulla aikayksiköllä. Punainen ja sininen käyrä ilmaisevat käytetyn ennustemallin sekä lineaarisen regression tekemiä ennusteita. Yleisesti ottaen, ennuste on kumulatiivisessa muodossaan tarkimmillaan ja heikoimillaan päiväkohtaisessa ennusteessa. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.', style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
                               ]
        return prediction, predict_placeholder_children,frequency_options, button_children

    
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
    elif rs == 'W':
        return plot_weekly_prediction(df)
    elif rs == 'M':
        return plot_monthly_prediction(df)
    elif rs == 'Q':
        return plot_quaterly_prediction(df)
    elif rs == 'Y':
        return plot_yearly_prediction(df)
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
    elif rs == 'W':
        return plot_weekly_test(df)
    elif rs == 'M':
        return plot_monthly_test(df)
    elif rs == 'Q':
        return plot_quaterly_test(df)
    elif rs == 'Y':
        return plot_yearly_test(df)
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
                                                 config=config_plots,
                                                figure = plot_daily_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa päivittäin.Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'W':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                 config=config_plots,
                                                figure = plot_weekly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa viikoittain. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'M':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                 config=config_plots,
                                                figure = plot_monthly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa kuukausittain. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'Q':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                 config=config_plots,
                                                figure = plot_quaterly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa kvartaaleittain. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
    elif rs == 'Y':
        return [dbc.Card(dbc.CardBody([dcc.Graph(id = 'original_graph',
                                                 config=config_plots,
                                                figure = plot_yearly_data(kunta,label))]),
                        body=False, 
                        color = 'dark'),
                html.Br(),
                html.P('Tässä kuvaajassa voit tarkastella valittua muuttujaa vuosittain. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
                       style={'textAlign':'center','font-family':'Arial', 'font-size':20, 'color':'black'})
               ]
     
    




@app.callback(

    Output('cumulative_graph_col', 'children'),
    
    [Input('cum_data_store', 'data'),
     Input('label_dropdown', 'value')
    ]

)
def update_cum_col(df, label):
    
    figure = plot_cum_data(df, label)
    
    return html.Div([dbc.Card(dcc.Graph(id = 'cumulative_graph',
                                        config=config_plots,
                                        figure = figure),
                              color='dark',
                              body=True),
                    html.Br(),
                    html.P('Tässä kuvaajassa voit tarkastella valitun muuttujan kumulatiivista kehitystä. Kuvaajan kuvioita voi piilottaa ja palauttaa klikkaamalla selitteen arvoja. Kuvaajan oikeassa yläkulmassa on työkalurivi, josta saa työkaluja muun muassa zoomaamiseen ja kuvatiedoston vientiin.',
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
        used_label = labels[label]
        kunta = prediction.kunta.values[0]
        pituus = prediction.ennusteen_pituus.values[0]
        reg_type = prediction.regularisointi.values[0]
        pr = prediction.copy()
        pr.drop(['name','kunta','ennusteen_pituus','regularisointi'],axis=1, inplace = True)
        pr = pr.rename(columns={'forecast':'Ennuste/Toteutunut',
                               'daily':'Päiväkohtainen arvo',
                                'first_pay_day_distance': 'Käänteinen etäisyys 1. maksupäivään',
                                'second_pay_day_distance': 'Käänteinen etäisyys 2. maksupäivään',
                                'third_pay_day_distance': 'Käänteinen etäisyys 3. maksupäivään',
                                'fourth_pay_day_distance': 'Käänteinen etäisyys 4. maksupäivään',
                                'edellinen':'Edellisen päivän arvo',
                                used_label:label,
                               'baseline': 'Ennuste (Lineaariregressio)',
                               'daily_baseline':'Päiväkohtainen LR-ennuste'})
        pr.index = [date.split()[0] for date in pr.index.astype(str)]
        pr.index.name = 'Maksupäivä'
        
        train_val_test_ = train_val_test.copy()

        train_val_test_ = train_val_test_.rename(columns={
                               'daily_true':'Päiväkohtainen arvo',
                                'daily_pred':'Päiväkohtainen ennuste',
                                'daily_baseline':'Päiväkohtainen LR-ennuste',
                                'first_pay_day_distance': 'Käänteinen etäisyys 1. maksupäivään',
                                'second_pay_day_distance': 'Käänteinen etäisyys 2. maksupäivään',
                                'third_pay_day_distance': 'Käänteinen etäisyys 3. maksupäivään',
                                'fourth_pay_day_distance': 'Käänteinen etäisyys 4. maksupäivään',
                                'edellinen':'Edellisen päivän arvo',
                                used_label:label,
                                'ennuste_edellinen':'Edellisen päivän ennuste',
                                'baseline_edellinen':'Edellisen päivän LR-ennuste',
                                'ennustettu': 'Ennuste ({})'.format(reg_type),
                               'baseline': 'Ennuste (Lineaariregressio)'})

        
        train_val_test_.index = [date.split()[0] for date in train_val_test_.index.astype(str)]
        train_val_test_.index.name = 'Maksupäivä'
                
        train_data = train_val_test_[train_val_test_.split=='train']
        val_data = train_val_test_[train_val_test_.split=='val']
        test_data = train_val_test_[train_val_test_.split=='test']
        
        alpha = test_data.alpha.values[0]
        split_portion = test_data.split_portion.values[0]
        
        remove = ['Päiväkohtainen arvo', 
                  'Päiväkohtainen ennuste',
                  'Päiväkohtainen LR-ennuste', 
                  'Edellisen päivän ennuste', 
                  'Edellisen päivän LR-ennuste',
                  'reg_type',
                  'name',
                  'kunta',
                 'alpha',
                  'split',
                  'label',
                  'Ennuste (Lineaariregressio)',
                  'Ennuste ({})'.format(reg_type),
                 'split_portion']
        
        train_data.drop(remove, axis=1, inplace=True)
        val_data.drop(remove, axis=1, inplace=True)

        
        test_data.drop(['split_portion', 'alpha','split','reg_type','name','kunta','label'],axis=1, inplace=True)
               
        
        test_data['Absoluuttinen virhe'] = np.absolute(test_data[label] - test_data['Ennuste ({})'.format(reg_type)])        
        test_data['Suhteellinen virhe'] = test_data['Absoluuttinen virhe'] / test_data[label]
        
        test_data['Absoluuttinen virhe (LR)'] = np.absolute(test_data[label] - test_data['Ennuste (Lineaariregressio)'])        
        test_data['Suhteellinen virhe (LR)'] = test_data['Absoluuttinen virhe (LR)'] / test_data[label]
        
        test_data['Absoluuttinen päivävirhe'] = np.absolute(test_data['Päiväkohtainen arvo'] - test_data['Päiväkohtainen ennuste'])        
        test_data['Suhteellinen päivävirhe'] = test_data['Absoluuttinen päivävirhe'] / test_data['Päiväkohtainen arvo']
        
        test_data['Absoluuttinen päivävirhe (LR)'] = np.absolute(test_data['Päiväkohtainen arvo'] - test_data['Päiväkohtainen LR-ennuste'])        
        test_data['Suhteellinen päivävirhe (LR)'] = test_data['Absoluuttinen päivävirhe (LR)'] / test_data['Päiväkohtainen arvo']        
        
                                                                                              
        mae = test_data['Absoluuttinen virhe'].mean()   
        mape = test_data['Suhteellinen virhe'].mean()
        accuracy = 1 - mape
        
        mae_lr = test_data['Absoluuttinen virhe (LR)'].mean()
        mape_lr = test_data['Suhteellinen virhe (LR)'].mean()
        accuracy_lr = 1 - mape_lr
        
        mae_day = test_data['Absoluuttinen päivävirhe'].mean()   
        mape_day = test_data['Suhteellinen päivävirhe'].mean()
        accuracy_day = 1 - mape_day
        
        mae_lr_day = test_data['Absoluuttinen päivävirhe (LR)'].mean()
        mape_lr_day = test_data['Suhteellinen päivävirhe (LR)'].mean()
        accuracy_lr_day = 1 - mape_lr_day
        
        
        metadata = pd.DataFrame([{'Kunta':kunta,
                                 'Suure': label,
                                 'Testi/validointiosuus': str(split_portion)+' päivää',
                                 'Regularisointisuure':alpha,
                                  'Regularisointityyppi': reg_type,
                                  'Testidatan pituus': str(len(test_data))+' päivää',
                                  'Testin keskimääräinen absoluuttinen virhe ({})'.format(reg_type): '{:,}'.format(round(mae,2)).replace(',',' ')+' €',
                                  'Testin keskimääräinen suhteellinen virhe ({})'.format(reg_type): str(round(100*mape,2))+' %',
                                  'Testin keskimääräinen tarkkuus ({})'.format(reg_type): str(round(100*accuracy,2))+' %',
                                  'Testin keskimääräinen absoluuttinen virhe (LR)': '{:,}'.format(round(mae_lr,2)).replace(',',' ')+' €',
                                  'Testin keskimääräinen suhteellinen virhe (LR)': str(round(100*mape_lr,2))+' %',
                                  'Testin keskimääräinen tarkkuus (LR)': str(round(100*accuracy_lr,2))+' %',

                                  'Testin keskimääräinen absoluuttinen päivävirhe ({})'.format(reg_type): '{:,}'.format(round(mae_day,2)).replace(',',' ')+' €',
                                  'Testin keskimääräinen suhteellinen päivävirhe ({})'.format(reg_type): str(round(100*mape_day,2))+' %',
                                  'Testin keskimääräinen päiväkohtainen tarkkuus ({})'.format(reg_type): str(round(100*accuracy_day,2))+' %',
                                  'Testin keskimääräinen absoluuttinen päivävirhe (LR)': '{:,}'.format(round(mae_lr_day,2)).replace(',',' ')+' €',
                                  'Testin keskimääräinen suhteellinen päivävirhe (LR)': str(round(100*mape_lr_day,2))+' %',
                                  'Testin keskimääräinen päiväkohtainen tarkkuus (LR)': str(round(100*accuracy_lr_day,2))+' %',                                  
                                  
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
#Run app.
if __name__ == "__main__":
    app.run_server(debug=in_dev, threaded = True)
