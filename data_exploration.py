import pandas as pd
import geopandas as gpd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import time
import folium
from branca.colormap import LinearColormap
import streamlit as st
import joblib


#root='streamlit-compagnon-immo'
root="."


@st.cache_data
def load_data_explor():
    t=time.time()
    df=pd.read_csv(f"{root}/data/processed/sirene/sirene_group_dviz_streamlit.csv.zip", sep=",", dtype={"code_dep": str}, compression="zip")
    print(f'time data load explor: {time.time()-t}')    

    return df

group = load_data_explor()

@st.cache_data
def build_plot_nb_crea_idf():
    axe_temps="annee_mois_creation"
    axe="Lib_division"
    axes=["lib_trancheEffectif"]#,"Lib_section", "Lib_division"]

    _group=group[pd.to_datetime(group["annee_mois_creation"]).dt.year>=2000]
    #courbe totale
    group_tranches = _group.groupby([axe_temps]).agg({'nb_etabl_crees' : 'sum'}).reset_index()
    fig_obs = px.line(group_tranches, x=axe_temps, y='nb_etabl_crees', title="Nombre de créations d'établissements" )

    
    plots=[fig_obs]

    for axe in axes:
        group_tranches = _group.groupby([axe,axe_temps]).agg({'nb_etabl_crees' : 'sum'}).reset_index()
        fig_obs = px.line(group_tranches, x=axe_temps, y='nb_etabl_crees', color=axe, title="Nombre de créations par tranches d'effectifs salariés")
        plots.append(fig_obs)


    axe="Lib_section"
    group_tranches_absent = _group[_group["lib_trancheEffectif"]=="Absent"].groupby(axe).agg({'nb_etabl_crees' : 'sum'}).reset_index().sort_values(by="nb_etabl_crees", ascending=False)
    fig_obs = px.bar(group_tranches_absent.head(10), y=axe, x='nb_etabl_crees', color=axe, title="Nombre de créations par secteur d'activité, sur la tranche d'effectifs 'Absent'", orientation='h',)
    
    plots.append(fig_obs)


    axe="code_dep"
    axe_alias="Depart."
    dict_dep={
        "75": "Paris", "92": "Hauts de Seine", "93": "Seine St Denis", "78": "Yvelines",
        "94": "Val de Marne", "91": "Essone", "95": "Va d'oise", "77": "Seine et Marne"}
    group_tranches_absent = _group[_group["lib_trancheEffectif"]=="Absent"].groupby(axe).agg({'nb_etabl_crees' : 'sum'}).reset_index().sort_values(by="nb_etabl_crees", ascending=False)
    group_tranches_absent["Depart."]= group_tranches_absent[axe].map(dict_dep)
    fig_obs = px.bar(group_tranches_absent, y=axe_alias, x='nb_etabl_crees', color=axe_alias, title="Nombre de créations par département, sur la tranche d'effectifs 'Absent'", orientation='h',)
    
    plots.append(fig_obs)

    return plots