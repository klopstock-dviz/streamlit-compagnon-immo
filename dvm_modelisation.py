
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from branca.colormap import LinearColormap
import folium

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.saving import load_model
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

import random

import warnings
warnings.filterwarnings('ignore')


root = 'streamlit'
#root = '.'

# Liste des zones d'emploi
df_zones_dvm = pd.read_csv(f'{root}/data/processed/dvm/dvm_par_zone/zone_name.csv')
path_dvm = f'{root}/data/processed/dvm/dvm_par_zone'
path_models_dvm = f'{root}/data/processed/dvm/models'

# Récupération map nombre de communes par zone d'emploi
geo_zone_emploi_dvm = gpd.read_file(f'{root}/data/referentiels/geojson/zone_emploi_idf_normandie.geojson')
geo_zone_emploi_dvm = geo_zone_emploi_dvm.set_index('libze2020')

# Récupération map nombre de communes par zone d'emploi
geo_zone_emploi_indicateurs_dvm = gpd.read_file(f'{root}/data/referentiels/geojson/zone_emploi_indicateurs_ratio.geojson')
geo_zone_emploi_indicateurs_dvm = geo_zone_emploi_indicateurs_dvm.set_index('libze2020')

# Récupération map nombre indicateur durée de vie moyenne entre 2011-2015 et 2016-2020
geo_dvm_2020_dvm = gpd.read_file(f'{path_dvm}/geo_dvm_communes_idf_normandie_2020.geojson')
geo_dvm_2020_dvm = geo_dvm_2020_dvm.set_index('nom')

# renvoi le dataframe de la durée de vie moyenne d'une zone d'emploi
def get_dvm_per_zone(zone_emploi) :
    df_zone = pd.read_csv(f'{path_dvm}/dvm_{zone_emploi}.csv')
    df_zone = df_zone.ffill()
    return df_zone

# Renvoi une liste de toutes les zones d'emploi
def get_list_zone_emploi():
    return df_zones_dvm['zone_name'].value_counts().reset_index()['zone_name'].to_list()

# Renvoi un pointeur d'affichage des durées de vie pour les zones d'emploi selectionnées
def display_dvm(selected) :

    list_dvm = []
    for zone_emploi in selected :
        df = pd.read_csv(f'{path_dvm}/dvm_{zone_emploi}.csv')
        df = df.rename(columns={"duree_observation_etab": zone_emploi, "dateDebut" : "Années"})

        if 'df_final' in locals():
            df_final = df_final.merge(df, on='Années', how='inner')
        else :
            df_final = df.copy()
    
    fig = px.line(df_final, x='Années', y=selected)
    
    return fig

# Affiche une carte des communes et des données d'évolution de la durée de vie moyenne
def display_map_dvm() :

    fig = px.choropleth(geo_zone_emploi_dvm,
                   geojson=geo_zone_emploi_dvm.geometry,
                   locations=geo_zone_emploi_dvm.index,
                   color="nb_com",
                   color_continuous_scale="Viridis",
                   projection="mercator")
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_geos(fitbounds="locations", visible=False)

    return fig
      
# Affiche les données associées au ratio la durée de vie moyenne entre 2011-2015 et 2016-2020
def display_dvm_2020(pow) :    

    indicateur ="indicateur_dvm"
    range_color=[0.5,1.5]
    if pow == 2 :
        indicateur = "indicateur_dvm_carre"
        range_color=[0.8,1.2]


    fig = px.choropleth(geo_dvm_2020_dvm,
                   geojson=geo_dvm_2020_dvm.geometry,
                   locations=geo_dvm_2020_dvm.index,
                   color=indicateur,
                   range_color=range_color,
                   color_continuous_scale="viridis",
                   projection="mercator")
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_geos(fitbounds="locations", visible=False)

    return fig

# Affiche les prédictions
def display_lstm_bi_prediction(zone_emploi, n_forecast, sequence_type='lowseq') :

    # Récupération du modèle
    loaded_model = load_model(f'{path_models_dvm}/rnn_lstm_bi_{sequence_type}_{zone_emploi}.keras')
    df_dvm = get_dvm_per_zone(zone_emploi)

    if sequence_type == 'lowseq' :
        sequence_length = 5
    else :
        sequence_length = 12 

    __, X_test, __, __, __ = build_scaled_train_test(df_dvm['duree_observation_etab'], n_forecast, sequence_length)

    X_test = X_test[:n_forecast]
    y_pred = loaded_model.predict(X_test)
    fig = display_prediction_over_test(df_dvm, y_pred, n_forecast)

    return fig

# Affiche les prédictions sarima log
def display_sarima_log(zone_emploi, n_forecast, type) :

    # Récupération du modèle
    loaded_model = load(f'{path_models_dvm}/sarima_{type}_{zone_emploi}')
    df_dvm = get_dvm_per_zone(zone_emploi)
    
    y_pred = loaded_model.predict(n_periods=n_forecast).values
    fig = display_prediction_over_test(df_dvm, y_pred, n_forecast)

    return fig

# Affichage du forecast de chacun des deux modèles 
def display_model_forecasting(zone_emploi, n_forecast) :
    
    df_dvm = get_dvm_per_zone(zone_emploi)
    sequence_length = 12

    # Récupération du modèle sarima
    loaded_model_sarima = load(f'{path_models_dvm}/sarima_full_log_{zone_emploi}')
    y_pred_sarima = loaded_model_sarima.predict(n_periods=n_forecast).values

    # Prédictions RNN
    __, y_pred_lstm_bi = prediction(n_forecast, zone_emploi, sequence_length, df_dvm, path=f'{root}/data/processed/dvm/models', model_prefix='rnn_model_')

    df_past = df_dvm
    df_past = df_past.rename(columns={"duree_observation_etab": "values"})
    #y_test = np.array(df_duree_moyenne[df_past.shape[0]:(df_past.shape[0]+n_forecast)]['duree_observation_etab'])
        
    df_future = pd.DataFrame(columns=['dateDebut','values'])
    df_future['dateDebut'] = pd.date_range(start=df_past['dateDebut'].iloc[-1], freq ='M', periods=n_forecast+1)
    df_future.drop(0, inplace=True)
    df_future['dateDebut'] = pd.to_datetime(df_future['dateDebut'])
    df_future['dateDebut'] = df_future['dateDebut'].dt.strftime('%Y-%m')
    df_future['values'] = y_pred_sarima

    df_test = pd.DataFrame(columns=['dateDebut','values'])
    df_test['dateDebut'] = pd.date_range(start=df_past['dateDebut'].iloc[-1], freq ='M', periods=n_forecast+1)
    df_test.drop(0, inplace=True)
    df_test['dateDebut'] = pd.to_datetime(df_future['dateDebut'])
    df_test['dateDebut'] = df_test['dateDebut'].dt.strftime('%Y-%m')
    df_test['values'] = y_pred_lstm_bi

    trace = go.Scatter(
        x = df_past['dateDebut'],
        mode='lines',
        y=df_past['values'],
        marker=dict(color='rgba(12, 124, 32, 0.5)'),
        name='duree moyenne'
        )  
        
    trace1 = go.Scatter(
            x = df_future['dateDebut'],
            mode='lines',
            y=df_future['values'],
            marker=dict(color='blue'),
            name='Forecast sarima'
        )

    trace2 = go.Scatter(
            x = df_test['dateDebut'],
            mode='lines',
            y=df_test['values'],
            marker=dict(color='orange'),
            name='Forecast LSTM'
        )

    layout = dict(title=f'Forecast sur {n_forecast} mois')
    data = [trace, trace1, trace2]
    fig = dict(data=data, layout=layout)

    return fig    


# Ajout des compsantes fourier cosinus et sinus en features
def _add_time_features(data, period):
    time_steps = np.arange(len(data))
    sin_feature = np.sin(2 * np.pi * time_steps / period)
    cos_feature = np.cos(2 * np.pi * time_steps / period)
    return np.hstack((data.reshape(-1, 1), sin_feature.reshape(-1, 1), cos_feature.reshape(-1, 1)))

# Arrangement des données en séquence 
def _create_supervised_sequences(data, sequence_length):
    """
    Crée des séquences supervisées avec une longueur de séquence donnée.
    
    :param data: Données augmentées avec les colonnes [valeur originale, sin, cos]
    :param sequence_length: Longueur des séquences (par exemple 12 pour une séquence de 12 mois)
    :return: Tuple (X, y) où X sont les séquences et y les valeurs futures correspondantes
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Séquence d'entrée (les 'sequence_length' valeurs précédentes)
        X.append(data[i:i + sequence_length])
        
        # Valeur à prédire (la valeur juste après la séquence) - 1ère colonne uniquement
        y.append(data[i + sequence_length, 0])  # La 1ère colonne contient la valeur d'origine

    return np.array(X), np.array(y)

# private : prédictions recursives sur boucle fermée
def _recursive_prediction(model, initial_sequence, future_steps, data, period=12):

    """
    Effectue des prédictions récursives pour plusieurs étapes dans le futur.
    
    :param model: Modèle LSTM entraîné pour prédire un pas de temps
    :param initial_sequence: La dernière séquence connue de données (shape: (sequence_length, num_features))
    :param future_steps: Nombre de pas dans le futur à prédire
    :param period: Période pour recalculer les composantes sin et cos (ex: 12 pour la saisonnalité annuelle)
    :return: Liste des valeurs prédites dans le futur
    """
    predictions = []
    current_sequence = initial_sequence.copy()
    steps = len(data)
    print(f'data lenght = {steps}')

    for i in range(future_steps):
        # Prédire la prochaine valeur
        next_value = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)[0, 0]
        
        # Ajouter cette prédiction à la liste des prédictions
        predictions.append(next_value)
   
        # Calculer les nouvelles composantes sin et cos pour la prochaine étape
        steps += 1
   
        sin_feature = np.sin(2 * np.pi * (steps) / period)
        cos_feature = np.cos(2 * np.pi * (steps) / period)
        
        # Créer la nouvelle entrée (nouvelle valeur prédite + sin et cos recalculés)
        next_input = np.array([next_value, sin_feature, cos_feature])
        
        # Mettre à jour la séquence (décaler et ajouter la nouvelle prédiction en fin de séquence)
        current_sequence = np.vstack((current_sequence[1:], next_input))
    
    return predictions

# Prédiction des valeurs en mode "Many to one" sur boucle fermée ( pas de réentrainement après chaque prévision) 
def prediction(future_steps, zone_emploi, sequence_length, df_dvm, path=f'{root}/data/processed/dvm/models', model_prefix='rnn_model_') :

    loaded_model = load_model(f'{path}/{model_prefix}{zone_emploi}.keras')
    
    data = df_dvm['duree_observation_etab'].values
    mean_base = data.reshape(-1, 1)[-future_steps:].flatten().mean()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    X_with_sin_cos = _add_time_features(data, period=12)
    X_sequences, __ = _create_supervised_sequences(X_with_sin_cos, sequence_length)

    predictions = _recursive_prediction(loaded_model, X_sequences[-1], future_steps, data, 12)
      
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(np.array(predictions))

    indicateur = np.array(predictions).mean() / mean_base
 
    return indicateur, predictions


# Transformation des données et split ( normalisation )
def build_scaled_train_test(serie, forecast, LOOK_BACK) :

    X1, y1 = df_to_X_y(serie, LOOK_BACK)
    shift = (len(serie) - forecast) - LOOK_BACK

    X_train, y_train = X1[:shift], y1[:shift]
    X_test, y_test = X1[shift:], y1[shift:]

    # Reshape des tableaux avant la normalisation 
    X_train = X_train.reshape(X_train.shape[0], LOOK_BACK)
    X_test = X_test.reshape(X_test.shape[0], LOOK_BACK)

    # Normalisation des données 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape des données normalisées en matrice
    X_train = X_train_scaled.reshape(X_train.shape[0], LOOK_BACK, 1)
    X_test = X_test_scaled.reshape(X_test.shape[0], LOOK_BACK, 1)

    return X_train, X_test, y_train, y_test, scaler

# Transformation en données supervisées ( lag )
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

# Affichage des prédisctions par dessus les données de tests ( réelles)
def display_prediction_over_test(df_duree_moyenne, y_pred, n_forecast) :

    df_past = df_duree_moyenne[0:-60]
    df_past = df_past.rename(columns={"duree_observation_etab": "values"})
    y_test = np.array(df_duree_moyenne[df_past.shape[0]:(df_past.shape[0]+n_forecast)]['duree_observation_etab'])

        
    df_future = pd.DataFrame(columns=['dateDebut','values'])
    df_future['dateDebut'] = pd.date_range(start=df_past['dateDebut'].iloc[-1], freq ='M', periods=n_forecast+1)
    df_future.drop(0, inplace=True)
    df_future['dateDebut'] = pd.to_datetime(df_future['dateDebut'])
    df_future['dateDebut'] = df_future['dateDebut'].dt.strftime('%Y-%m')
    df_future['values'] = y_pred

    df_test = pd.DataFrame(columns=['dateDebut','values'])
    df_test['dateDebut'] = pd.date_range(start=df_past['dateDebut'].iloc[-1], freq ='M', periods=n_forecast+1)
    df_test.drop(0, inplace=True)
    df_test['dateDebut'] = pd.to_datetime(df_future['dateDebut'])
    df_test['dateDebut'] = df_test['dateDebut'].dt.strftime('%Y-%m')
    df_test['values'] = y_test

    trace = go.Scatter(
        x = df_past['dateDebut'],
        mode='lines',
        y=df_past['values'],
        marker=dict(color='rgba(12, 124, 32, 0.5)'),
        name='duree moyenne'
        )  
        
    trace1 = go.Scatter(
            x = df_future['dateDebut'],
            mode='lines',
            y=df_future['values'],
            marker=dict(color='blue'),
            name='duree moyenne predite'
        )

    trace2 = go.Scatter(
            x = df_test['dateDebut'],
            mode='lines',
            y=df_test['values'],
            marker=dict(color='red'),
            name='duree moyenne réelle'
        )

    layout = dict(title=f'Prédictions sur {n_forecast} mois - comparaison avec données de test')
    data = [trace, trace1, trace2]
    fig = dict(data=data, layout=layout)

    return fig

# Affiche les métriques RMSE, MAPE, RATIO pour chacun des deux modèles sur l'ensemble des zons d'emplois
def display_metrics_dvm() :
    df_metrics_dvm = pd.read_csv(f'{path_dvm}/metrics_evaluation_lstm_sarima.csv')

    plt.figure(figsize=(15,6))
    df_melted = pd.melt(df_metrics_dvm, ['ze'])
    fig = px.line(data_frame=df_melted, x='ze', y='value', color='variable')
    
    return fig

def display_boxplot_ratio() :
    df_metrics_dvm = pd.read_csv(f'{path_dvm}/metrics_evaluation_lstm_sarima.csv')
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_metrics_dvm['sarima_ratio'],  name="ratio moyen sarima"))
    fig.add_trace(go.Box(y=df_metrics_dvm['lstm_bi_ratio'], name="ratio moyen LSTM"))
    fig.update_traces(boxpoints='all', jitter=0.3)
    return fig

def display_boxplot_mape() :
    df_metrics_dvm = pd.read_csv(f'{path_dvm}/metrics_evaluation_lstm_sarima.csv')
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_metrics_dvm['sarima_mape'],  name="MAPE sarima"))
    fig.add_trace(go.Box(y=df_metrics_dvm['lstm_bi_mape'], name="MAPE LSTM"))
    fig.update_traces(boxpoints='all', jitter=0.3)
    return fig


def display_boxplot_rmse() :
    df_metrics_dvm = pd.read_csv(f'{path_dvm}/metrics_evaluation_lstm_sarima.csv')
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_metrics_dvm['sarima_rmse'],  name="RMSE sarima"))
    fig.add_trace(go.Box(y=df_metrics_dvm['lstm_bi_rmse'], name="RMSE LSTM"))
    fig.update_traces(boxpoints='all', jitter=0.3)
    return fig





# Affiche une carte folium avec les zones d'emplois pondérés par le champs field 
def build_map_zone_emploi(field):   

    df = geo_zone_emploi_indicateurs_dvm
    df = df.reset_index()
    df = df.rename(columns={'libze2020': 'Zone'})

    if (field=="sarima_ratio" or field=="lstm_bi_ratio") :
        # Custom diverging color map (Green at 1, red below and above)
        custom_colormap = LinearColormap(
            colors=['red', 'orange', 'green','orange', 'red'],  # Red for <1, Green at 1, Yellow for >1
            #vmin=df['ratio'].min(),  # Minimum value of your 'ratio' column
            vmin=0.5,
            vmax=1.5,
            caption='Ratio (vert = meilleur)'
        ).scale(0.5, 1.5)
    elif (field=="indicateur_forecast_sarima" or field=='indicateur_forecast_sarima_carre' or field=='indicateur_forecast' or field=='indicateur_forecast_carre') :
        custom_colormap = LinearColormap(
            colors=['black', 'red','orange', 'yellow', 'green', 'green', 'green',  'green'],  # Red for <1, Green at 1, Yellow for >1            
            #vmin=df[field].min(),  # Minimum value of your 'ratio' column,
            #vmax=df[field].max(),  # Minimum value of your 'ratio' column
            vmin=0.8,  # Minimum value of your 'ratio' column,
            vmax=1.4,  # Minimum value of your 'ratio' column
            caption='Vert = meilleur'
        ).scale(0.8, 1.4)
        
    # Create a folium map centered around your geometries
    m = folium.Map(
        location=[48.91, 0.71], 
        zoom_start=8, prefer_canvas=True)

    # Function to style each feature (polygon) based on 'ratio' value
    def style_function(feature):
        field_value = feature['properties'][field]
        return {
            'fillColor': custom_colormap(field_value),  # Get color from colormap
            'color': 'white',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }

    # Add the GeoJson layer to the map, including the ratio for styling
    folium.GeoJson(
        df,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['Zone', field])  # Display additional fields in tooltips
    ).add_to(m)

    # Get the bounds (bounding box) of the GeoDataFrame to fit the map to the geometry
    bounds = df.total_bounds  # [minx, miny, maxx, maxy]

    # Fit the map to the bounds of the geometry (bbox)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Add the custom colormap to the map as a legend
    custom_colormap.add_to(m)

    return m