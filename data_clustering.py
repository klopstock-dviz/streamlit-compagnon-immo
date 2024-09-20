import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from umap import UMAP
import time
import folium
import branca.colormap as cm
import streamlit as st


#root='streamlit-compagnon-immo'
root="."

@st.cache_data
def load_data_clustering():
    t=time.time()
    df_stats_communes=pd.read_csv(f"{root}/data/processed/clustering/df_stats_communes.csv.zip", sep=",", dtype={"CODGEO": str, "REG": str}, compression="zip")
    df_communes=pd.read_csv(f"{root}/data/referentiels/cog_ensemble_2023_csv/v_commune_2023.csv.zip", sep=',', dtype={"COM": str, "REG": str}, compression="zip")
    print("data clustering load time", time.time()-t)

    # com_geo = gpd.GeoDataFrame([])
    # for reg in ["idf", "normandie"]:
    #     cg=gpd.read_file(ff"{root}/data/referentiels/geojson/communes-{reg}.geojson")
    #     com_geo=pd.concat([com_geo, cg], axis=0)

    # com_geo.drop_duplicates(subset=['code'], inplace=True)

    # com_geo['geometry'] = com_geo['geometry'].simplify(tolerance=0.002)


    #com_geo.to_file(f"{root}/data/processed/communes_processed.geojson", driver="GeoJSON")

    t=time.time()
    com_geo = gpd.read_file(f"{root}/data/referentiels/geojson/communes_processed.geojson.zip", compression="zip")
    print(f"geojson load time: {time.time()-t}")

    return df_stats_communes, df_communes, com_geo

df_stats_communes, df_communes, com_geo= load_data_clustering()


@st.cache_data
def load_shapefile_ZE():
    t=time.time()
    ZE_geo = gpd.read_file(f"{root}/data/referentiels/ze2020_2023/ze2020_2023.shp",)
    ZE_communes = pd.read_csv(f"{root}/data/referentiels/ze2020_2023/zones_emloi.csv", dtype={"CODGEO": str, "ZE2020": str, "DEP": "str",})
    ZE_communes=ZE_communes[ZE_communes["REG"].isin([11, 28])] # filtre sur IDF & Nomrandie
    ZE_geo=ZE_geo.merge(ZE_communes, left_on="ze2020", right_on="ZE2020", how="inner")
    ZE_geo.drop(columns=["CODGEO","LIBGEO","ZE2020", "LIBZE2020", "ZE2020_PARTIE_REG","DEP"], inplace=True)
    ZE_geo=ZE_geo[ZE_geo["REG"].isin([11, 28])]
    ZE_geo.drop_duplicates(inplace=True)
    print(f'time shapefile ZE volumes: {time.time()-t}')    
    return ZE_geo, ZE_communes

ZE_geo, ZE_communes=load_shapefile_ZE()

@st.cache_data
def init_dict_df_clusters():
    return {
        "PCA": [],
        'T-SNE': [],
        "UMAP": [],
        "ratios_moyens": []
    }

dict_df_clusters=init_dict_df_clusters()

def get_data_head_clustering():    
    return df_stats_communes.head()

def get_data_describe_clustering():
    return df_stats_communes.describe()

def build_hist_population_clustering():

# Define the bins
    bins = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, float('inf')]
    labels = ['0-500', '501-1k', '1k-2k', '2k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-200k', "200k+"]

    # Create a new column with the binned data
    df_stats_communes['population_category'] = pd.cut(df_stats_communes['P21_POP'], bins=bins, labels=labels, right=False)

    # Count the municipalities in each bin
    #population_distribution = pd.DataFrame(df_stats_communes['population_category'].value_counts().sort_index())
    population_distribution = df_stats_communes['population_category'].value_counts().sort_index().reset_index()

    return population_distribution

def get_columns_clustering():
    df_cols_clustering=pd.DataFrame(
        [
            {"field": 'densite_demographique', "alias": "Densité démo.", "Définition": '', "dimension": "Société", "active": True},
            {"field": 'tx_evol_demographique', "alias": "% évol. démo.","Définition": 'Taux de coissance entre 2015 et 2021', "dimension": "Société", "active": True},
            {"field": "ind_jeunesse", "alias": "Ind. de jeunesse", "dimension": "Société", "active": True},
            {"field": "tx_scolarisation", "alias": "Tx de scolarisation", "dimension": "Société", "active": False},
            {"field": "tx_pop_sans_diplome", "alias": "% population sans diplome", "dimension": "Société", "inverse": True, "active": True},
            {"field": "tx_pop_bac+2_ou_+", "alias": "% pop bac+2 ou +", "dimension": "Société", "active": True},
            {"field": 'TP6021', "alias": "Tx pauvreté", "Définition": "Concerne les personnes dont le revenu est inférieur à 60% du revenu médian annuel pour une année donnée (24 330 euros en 2022)", "dimension": "Société", "inverse": True, "active": True},
            {"field": 'MED21', "alias": "Niveau de vie median", "dimension": "Société", "Définition": 'Le niveau de vie est égal au revenu disponible du ménage divisé par le nombre d’unités de consommation (UC).', "active": True},
            {"field": 'tx_chomage', "alias": "Tx chômage", "dimension": "Société", "inverse": True, "active": False},
            {"field": 'prix_m2_logmts', "alias": "Prix m2 logts.", "dimension": "Immobilier", "active": True},
            {"field": '%_evol_nb_ventes_logmts', "alias": "%_evol_nb_ventes_logmts", "dimension": "Immobilier", "active": True},
            {"field": '%_evol_prix_m2_logmts', "alias": "%_evol_prix_m2_logmts", "dimension": "Immobilier", "active": True},
            {"field": 'tx_vacance_log', "alias": "Tx vacance log.", "dimension": "Immobilier", "inverse": True, "active": True},
            {"field": 'prix_m2_locaux', "alias": "Prix m2 locaux.", "dimension": "Economie", "active": True},
            {"field": '%_evol_nb_ventes_locaux', "alias": "%_evol_nb_ventes_locaux", "dimension": "Economie", "active": False},
            {"field": '%_evol_prix_m2_locaux', "alias": "%_evol_prix_m2_locaux", "dimension": "Economie", "active": False},
            {"field": "indicateur_dvm", "alias": "Ind. durée vie établ.", "dimension": "Economie", "active": True},
            {"field": "%_evol_creations_etabl", "alias": "Ind. créations établ.", "dimension": "Economie", "active": True},
            {"field": 'tx_activite', "alias": "Tx activité", "dimension": "Economie", "active": True},
            {"field": 'part_emplois_salaries', "alias": "% emplois salariés", "dimension": "Economie", "active": True},# discuter
            {"field": 'part_emplois_dans_la_commune', "alias": "% emplois dans la commune", "dimension": "Economie", "active": False},# discuter
            {"field": 'densite_etablissements', "alias": "Densité établ.", "dimension": "Economie", "active": True},# discuter
            {"field": 'tx_etablissements', "alias": "Tx établ.", "dimension": "Economie", "active": False},# discuter
            {"field": 'part_Ets_agriculture', "alias": "% établ. agriculture", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_industrie', "alias": "% établ. industrie", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_construction', "alias": "% établ. construction", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_commerce_services', "alias": "% établ. commerce_services", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_adm_publique', "alias": "% établ. adm_publique", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_1_a_9_salariés', "alias": "% établ. 1_a_9_salariés", "dimension": "Economie", "active": False},
            {"field": 'part_Ets_10_salaries_ou_+', "alias": "% établ. 10_salaries_ou_+", "dimension": "Economie", "active": False},
        ]        
    )
    return df_cols_clustering
# pre processing 

def build_correlations_clustering(thresh_corr_clustering):
    keep=[ '%_evol_creations_etabl',
       'indicateur_dvm', 'prix_m2_logmts', '%_evol_nb_ventes_logmts',
       '%_evol_prix_m2_logmts', 'prix_m2_locaux', '%_evol_nb_ventes_locaux',
       '%_evol_prix_m2_locaux', 'pop_0_29ans', 'ind_jeunesse',
       'tx_scolarisation', 'tx_pop_sans_diplome', 'tx_pop_bac+2_ou_+',
       'densite_demographique', 'tx_evol_demographique', 'tx_vacance_log',
       'tx_activite', 'tx_chomage', 'part_emplois_salaries',
       'part_emplois_dans_la_commune', 'densite_etablissements',
       'tx_etablissements',]
    df_corr=df_stats_communes[keep].select_dtypes(include="number").corr()
    df_corr=df_corr[(df_corr<=thresh_corr_clustering*-1)|(df_corr>=thresh_corr_clustering)].fillna(0)
    #df_corr= df_corr[((df_corr>1)& (df_corr<=thresh_corr_clustering*-1) )| ( (df_corr>=thresh_corr_clustering) & (df_corr<1) ) ].fillna(0)
    #fig = px.imshow(df_corr, text_auto=False, )
    fig = px.imshow(df_corr, 
                labels=dict(x="Variables", y="Variables", color="Correlation"),
                x=df_corr.columns,
                y=df_corr.columns,
                color_continuous_scale='RdBu_r',  # Customize the color scheme
                width=940,
                height=940
                #aspect=""
    )

    return fig

def inverse_values_clustering(df, columns_to_scale):
    for c in columns_to_scale:
        if 'inverse' in list(c.keys()) and c["inverse"]==True:
            df[c["field"]]=1-df[c["field"]]        
    return df

def scale_data_clustering(df, columns_to_scale=[]):

    cols_drop=[
        'CODGEO', 'REG', "P21_POP", "P15_POP", "SUPERF", "P21_MEN",
        "NAIS1520",
        "DECE1520",
        "NAISD22",
        "DECESD22",
        "P21_LOG",
        "P21_RP",
        "P21_RSECOCC",
        "P21_LOGVAC",
        "P21_RP_PROP",
        "NBMENFISC21",
        "P21_EMPLT",
        "P21_EMPLT_SAL",
        "P15_EMPLT",
        "P21_POP1564",
        "P21_CHOM1564",
        "P21_ACT1564",
        "ETTOT21",
        "ETAZ21",
        "ETBE21",
        "ETFZ21",
        "ETGU21",
        "ETGZ21",
        "ETOQ21",
        "ETTEF121",
        "ETTEFP1021",
        "pop_0_29ans",
        "population_category"
    ]
    X = df.drop(cols_drop, axis=1)

    scaler = StandardScaler()
    if len(columns_to_scale)>0:
        #inverse_values()
        columns_to_scale=[f["field"] for f in columns_to_scale]
        X_scaled = scaler.fit_transform(X[columns_to_scale])
    else:
        #inverse_values()
        X_scaled = scaler.fit_transform(X)

    return X_scaled



def select_subset_clustering(type_ind="all", population=1000):
    # scale df selon type d'indicateur et nb habitants
    """liste des indicateurs:    
        - Société
        - Immobilier
        - Economie
    """
    
    dict_ind=[
        {"field": 'densite_demographique', "alias": "Densité démo.", "dimension": "Société", "active": True},
        {"field": 'tx_evol_demographique', "alias": "% évol. démo.", "dimension": "Société", "active": True},
        {"field": "ind_jeunesse", "alias": "Ind. de jeunesse", "dimension": "Société", "active": True},
        {"field": "tx_scolarisation", "alias": "Tx de scolarisation", "dimension": "Société", "active": False},
        {"field": "tx_pop_sans_diplome", "alias": "% pop sans diplome", "dimension": "Société", "inverse": True, "active": True},
        {"field": "tx_pop_bac+2_ou_+", "alias": "% pop bac+2 ou +", "dimension": "Société", "active": True},
        {"field": 'TP6021', "alias": "Tx pauvreté", "dimension": "Société", "inverse": True, "active": True},
        {"field": 'MED21', "alias": "Niveau de vie median", "dimension": "Société", "active": True},
        {"field": 'tx_chomage', "alias": "Tx chômage", "dimension": "Société", "inverse": True, "active": False},
        {"field": 'prix_m2_logmts', "alias": "Prix m2 logts.", "dimension": "Immobilier", "active": True},
        {"field": '%_evol_nb_ventes_logmts', "alias": "%_evol_nb_ventes_logmts", "dimension": "Immobilier", "active": True},
        {"field": '%_evol_prix_m2_logmts', "alias": "%_evol_prix_m2_logmts", "dimension": "Immobilier", "active": True},
        {"field": 'tx_vacance_log', "alias": "Tx vacance log.", "dimension": "Immobilier", "inverse": True, "active": True},
        {"field": 'prix_m2_locaux', "alias": "Prix m2 locaux.", "dimension": "Economie", "active": True},
        {"field": '%_evol_nb_ventes_locaux', "alias": "%_evol_nb_ventes_locaux", "dimension": "Economie", "active": False},
        {"field": '%_evol_prix_m2_locaux', "alias": "%_evol_prix_m2_locaux", "dimension": "Economie", "active": False},
        {"field": "indicateur_dvm", "alias": "Ind. durée vie établ.", "dimension": "Economie", "active": True},
        {"field": "%_evol_creations_etabl", "alias": "Ind. créations établ.", "dimension": "Economie", "active": True},
        {"field": 'tx_activite', "alias": "Tx activité", "dimension": "Economie", "active": True},
        {"field": 'part_emplois_salaries', "alias": "% emplois salariés", "dimension": "Economie", "active": True},# discuter
        {"field": 'part_emplois_dans_la_commune', "alias": "% emplois dans la commune", "dimension": "Economie", "active": False},# discuter
        {"field": 'densite_etablissements', "alias": "Densité établ.", "dimension": "Economie", "active": True},# discuter
        {"field": 'tx_etablissements', "alias": "Tx établ.", "dimension": "Economie", "active": False},# discuter
        {"field": 'part_Ets_agriculture', "alias": "% établ. agriculture", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_industrie', "alias": "% établ. industrie", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_construction', "alias": "% établ. construction", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_commerce_services', "alias": "% établ. commerce_services", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_adm_publique', "alias": "% établ. adm_publique", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_1_a_9_salariés', "alias": "% établ. 1_a_9_salariés", "dimension": "Economie", "active": False},
        {"field": 'part_Ets_10_salaries_ou_+', "alias": "% établ. 10_salaries_ou_+", "dimension": "Economie", "active": False},
    ]




    if type_ind=='Tout':
        selected_inds=[r for r in dict_ind if r["active"]==True]
    else:
        selected_inds=[r for r in dict_ind if (r["dimension"]==type_ind and r["active"]==True)]


    df_stats_communes_filtre=df_stats_communes[df_stats_communes['P21_POP']>= population].reset_index().dropna()
    df_stats_communes_filtre=inverse_values_clustering(df_stats_communes_filtre, selected_inds)
    X_scaled=scale_data_clustering(df_stats_communes_filtre, selected_inds)
    
    return {"X_scaled": X_scaled, "df": df_stats_communes_filtre, "selected_inds": selected_inds}

def get_pca_components_clustering(X, verbose=False):
    # PCA
    #k
    pca=PCA()
    #l
    prj= pca.fit_transform(X)


    explained_variance= pd.Series(pca.explained_variance_ratio_.cumsum())
    n_components= explained_variance[explained_variance> 0.88].index[0]+1

    return n_components


def get_df_from_lower_dimension_clustering(X, df_source, cols, clusters):
    df=pd.DataFrame(data=X, columns=cols)
    df["region"]= df_source["REG"].map({"11": "IDF", '28': "Normandie"})
    df["CODGEO"]= df_source["CODGEO"]
    df["dep"]= df["region"]+"-"+ df_source["CODGEO"].str[:2]
    df['clusters'] = clusters.astype(str)
    df=df.merge(df_source, on='CODGEO', how='inner')
    df=df.merge(df_communes, left_on='CODGEO', right_on='COM', how='inner').drop(
        columns=[
            'TYPECOM', 'COM',
            'REG_y', 'DEP', 'CTCD', 'ARR', 'TNCC', 'NCC', 'NCCENR',
            'CAN', 'COMPARENT'
        ]
    ).rename(columns={'LIBELLE': 'libcom'})

    return df


# colors schemes: https://vega.github.io/vega/docs/schemes/
def get_altair_scatter_clustering(data, selected_inds, scatter_only=True):
    nb_clusters=len(data["clusters"].unique())

    brush = alt.selection_interval()
    scatter_width=960*0.85
    points = alt.Chart(data).mark_circle().encode(
        x=alt.Y('axe_0:Q', scale=alt.Scale(zero=False, padding=0)),  # Adjust y-axis scale
        y=alt.Y('axe_1:Q', scale=alt.Scale(zero=False, padding=0)),  # Adjust y-axis scale
        color=alt.condition(brush, alt.Color('clusters:N', scale=alt.Scale(scheme='category10')), alt.value('lightgray')),
        tooltip=['libcom:N', 'dep:N', 'region:N', 'clusters:N']
    ).add_params(
        brush
    ).properties(
        width=scatter_width,
        height=scatter_width*0.6,
        title=f"Communes en {nb_clusters} clusters"
    )#.configure_axis(
    #     labelColor='white',
    #     titleColor='white'
    # ).configure_view(
    #     strokeWidth=0,
    #     fill='white'
    # )

    bars_dep = alt.Chart(data).mark_bar().encode(
        y='dep:N',
        color='clusters:N',
        x='count(dep):Q'
    ).transform_filter(
        brush
    ).properties(width=225)

    bars_region = alt.Chart(data).mark_bar().encode(
        y='region:N',
        color='clusters:N',
        x='count(region):Q'
    ).transform_filter(
        brush
    ).properties(width=225)

    bars = alt.vconcat(bars_region, bars_dep).resolve_legend(
        color="independent"
    )


    # Create the boxplots
    boxplots_encodings=[]
    for e in selected_inds[:5]:
        boxplot = alt.Chart(data).mark_boxplot().encode(
            #x='category:N',
            x=alt.Y(f'{e["field"]}:Q', title=e["alias"])
        ).transform_filter(
            brush
        )#.properties(width=225)

        boxplots_encodings.append(boxplot)

    boxplots = alt.vconcat(*boxplots_encodings)

    bars_text = alt.hconcat(bars, boxplots).resolve_legend(
        color="independent"
    )

    # Combine scatter plot and data table
    if scatter_only:
        return {"final_chart": points, "points": points}
    elif scatter_only==False:
        final_chart = alt.vconcat(points, bars_text).resolve_legend(
            color="independent"
        ).configure_view(
            stroke=None
        )

        # Display the chart
        return {"final_chart": final_chart, "points": points}

def get_scatter_colorScheme_clustering(points, data):

    # Manually define the category10 colors (D3 color scheme)
    category10_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#1f77b4", "#ff7f0e",
    ]

    # Extract unique clusters and map them to the color scheme
    clusters = sorted(data['clusters'].unique())  # Sorted unique cluster labels
    color_mapping = {str(cluster): category10_colors[i] for i, cluster in enumerate(clusters)}

    return color_mapping

def get_altair_heatmap_clustering(df, indicateurs):

    fields=[f["field"] for f in indicateurs]
    group=df.groupby('clusters')[fields].mean().T
    # Reset index and melt the DataFrame
    group_melted = group.reset_index().melt(id_vars='index', var_name='Cluster', value_name='Value')

    # Calculate min and max for each characteristic
    group_min_max = group_melted.groupby('index').agg({'Value': ['min', 'max']}).reset_index()
    group_min_max.columns = ['index', 'min', 'max']

    # Merge min and max values back to the melted DataFrame
    group_merged = pd.merge(group_melted, group_min_max, on='index')

    # get aliases
    group_merged=group_merged.merge(pd.DataFrame(indicateurs), left_on='index', right_on='field', how="left").drop(columns=['field'])
    
    
    # Calculate scaled values
    group_merged['Scaled_Value'] = (group_merged['Value'] - group_merged['min']) / (group_merged['max'] - group_merged['min'])

    # Calculate the mean of Scaled_Value for each Cluster
    mean_values = group_merged.groupby('Cluster')['Scaled_Value'].mean().reset_index()

    # Add a new index for the mean values
    mean_values['index'] = 'Ratios moyens'

    # Append the mean values as a new row to the DataFrame
    group_merged = pd.concat([group_merged, mean_values], ignore_index=True)

    # Ensure that the mean_Scaled_Values column is updated accordingly
    group_merged['Ratios moyens'] = group_merged.groupby('Cluster')['Scaled_Value'].transform('mean')
    group_merged.loc[:, 'alias']= group_merged["alias"].fillna("Ratios moyens")
    
    # Create the heatmap using Altair
    
    # Custom sort order for the 'index' column, placing 'mean_Scaled_Values' last
    sort_order = group_merged['alias'].unique().tolist()
    sort_order.remove('Ratios moyens')
    sort_order.append('Ratios moyens')

    # Create the heatmap using Altair
    heatmap = alt.Chart(group_merged).mark_rect().encode(
        x=alt.X(
            'Cluster:O', 
            title='Cluster', 
            sort=alt.EncodingSortField(
                field='Ratios moyens',  # Sort by mean_Scaled_Values
                order='descending'  # Adjust this to 'ascending' if needed
            ),
            scale=alt.Scale(paddingInner=0.05)
        ),
        y=alt.Y(
            'alias:O', 
            title='Characteristics',
            sort=sort_order,  # Use the custom sort order
            scale=alt.Scale(paddingInner=0.05)
        ),
        color=alt.Color(
            'Scaled_Value:Q', 
            scale=alt.Scale(scheme='blues'), 
            legend=alt.Legend(title='Scaled Value')
        ),
        tooltip=[
            alt.Tooltip('alias:N', title='Indicateur'),
            alt.Tooltip('Cluster:N'),
            alt.Tooltip('Scaled_Value:Q', title='Scaled Value', format='.2f'),
            alt.Tooltip('Value:Q', title='Original Value', format='.2f')
        ]
    ).properties(
        title="Heatmap des clusters (mise à l'échelle Min-Max par indicateur)",
        width=400,
        height=600
    )

            
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y(
            'clusters:N', 
            sort='ascending'  # Sort descending to have the largest count at the top            
        ),
        x='count(axe_1):Q').properties(
        title="Nb de communes par cluster"
    )

    
    # Combine scatter plot and data table
    final_chart = alt.vconcat(heatmap, bars).resolve_legend(
        color="independent"
    ).configure_view(
        stroke=None
    )

    ratios_moyens=group_merged[group_merged["index"]=="Ratios moyens"][["Cluster", "Ratios moyens"]].reset_index(drop=True)
    
    return final_chart, ratios_moyens
    #return {"heatmap": final_chart, "ratios_moyens_clusters": ratios_moyens}


def build_pca_clustering(type_ind, population, n_clusters, scatter_only):
    
    data=select_subset_clustering(type_ind, population)
    X_scaled, df, selected_inds=data["X_scaled"], data["df"], data["selected_inds"]
    
    n_components=get_pca_components_clustering(X=X_scaled, verbose=False)

    pca_cols=[]
    for i in range(0,n_components):
        pca_cols.append(f'axe_{i}')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)    

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    df_pca= get_df_from_lower_dimension_clustering(X_pca, df, pca_cols, clusters)
    dviz= get_altair_scatter_clustering(df_pca, selected_inds, scatter_only)
    final_chart=dviz["final_chart"]
    
    

    heatmap, ratios_moyens=get_altair_heatmap_clustering(df_pca, selected_inds)
    
    #colorScheme=get_scatter_colorScheme(points, df_pca)
    #map=build_map_v3(population, df_pca, ratios_moyens,field)

    dict_df_clusters["PCA"]= df_pca
    dict_df_clusters["ratios_moyens"]= ratios_moyens

    return {"scatter": final_chart, "heatmap": heatmap,}    

def build_tsne_clustering(type_ind, population, n_clusters, scatter_only):
    tsne=TSNE(method="barnes_hut",)

    data=select_subset_clustering(type_ind, population)
    X_scaled, df, selected_inds=data["X_scaled"], data["df"], data["selected_inds"]

    X_tsne=tsne.fit_transform(X_scaled)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_tsne)

    df_tsne= get_df_from_lower_dimension_clustering(X_tsne, df, ["axe_0", "axe_1"], clusters)
    dviz= get_altair_scatter_clustering(df_tsne, selected_inds, scatter_only)
    final_chart=dviz["final_chart"]
    
    

    heatmap, ratios_moyens=get_altair_heatmap_clustering(df_tsne, selected_inds)
    
    #colorScheme=get_scatter_colorScheme(points, df_pca)
    #map=build_map_v3(population, df_tsne, ratios_moyens,)

    dict_df_clusters["T-SNE"]= df_tsne
    dict_df_clusters["ratios_moyens"]= ratios_moyens

    return {"scatter": final_chart, "heatmap": heatmap}  
    

def build_umap_clustering(type_ind, population, n_clusters, scatter_only):
    umap_model = UMAP()

    data=select_subset_clustering(type_ind, population)
    X_scaled, df, selected_inds=data["X_scaled"], data["df"], data["selected_inds"]

    X_umap = umap_model.fit_transform(X_scaled)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_umap)

    df_umap= get_df_from_lower_dimension_clustering(X_umap, df, ["axe_0", "axe_1"], clusters)
    dviz= get_altair_scatter_clustering(df_umap, selected_inds, scatter_only)
    final_chart=dviz["final_chart"]
    
    heatmap, ratios_moyens=get_altair_heatmap_clustering(df_umap, selected_inds)
    
    #colorScheme=get_scatter_colorScheme(points, df_pca)
    #map=build_map_v3(population, df_umap, ratios_moyens, field)

    dict_df_clusters["UMAP"]= df_umap
    dict_df_clusters["ratios_moyens"]= ratios_moyens    

    return {"scatter": final_chart, "heatmap": heatmap}  

def build_map_clustering_v1(population, df, field="clusters"):
    # m = folium.Map([45.35, -121.6972], zoom_start=12)
# 
    # folium.Marker(
        # location=[45.3288, -121.6625],
        # tooltip="Click me!",
        # popup="Mt. Hood Meadows",
        # icon=folium.Icon(icon="cloud"),
    # ).add_to(m)
   

    com_data = df[(df["P21_POP"]>=population)]

    # Merge the GeoJSON with your data
    #merged_data = com_data.merge(com_geo, left_on=['CODGEO'], right_on=['code'], how='inner')
    merged_data = com_geo.merge(com_data, left_on=['code'], right_on=['CODGEO'], how='inner')


    m = folium.Map(location=[48.71, 0.71], zoom_start=8, prefer_canvas=True)

    # Create a colormap
    merged_data["clusters"]=merged_data["clusters"].astype(int)
    
    n_clusters = len(merged_data['clusters'].unique())
    
    if field=='clusters':
        # Create a colormap using 'category10'
        n_clusters = len(merged_data['clusters'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78']  # category10 colors + 2 extra
        colormap = cm.LinearColormap(colors=colors[:n_clusters], 
            vmin=merged_data['clusters'].min(), 
            vmax=merged_data['clusters'].max()
        )
        colorScheme='Blues'
    else:
        # Generate blue color palette
        def generate_blue_palette(num_colors):
            return ['#{:02x}{:02x}{:02x}'.format(int(255 * (1 - i / (num_colors - 1))), 
                                                int(255 * (1 - i / (num_colors - 1))), 
                                                255) 
                    for i in range(num_colors)]

        blue_colors = generate_blue_palette(n_clusters)
        # Create a colormap
        colormap = cm.LinearColormap(
            colors=blue_colors,
            vmin=merged_data['clusters'].min(),
            vmax=merged_data['clusters'].max()
        )
        colorScheme='Blues'
        
    
    # Add the choropleth layer
    folium.Choropleth(
        geo_data=merged_data,
        name='choropleth',
        data=merged_data,
        columns=['CODGEO', field],  
        key_on='feature.properties.code',
        fill_color=colorScheme,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=field  
    ).add_to(m)

    # Add hover functionality without creating a new legend
    style_function = lambda x: {'fillColor': 'transparent', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    folium.GeoJson(
        merged_data,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['nom', field],
            aliases=['Commune:', f'{field}:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    # Add color scale
    #colormap.add_to(m)


    return m


def build_map_clustering_v2(population, df, field):

    com_data = df[(df["P21_POP"]>=population)]

    # Merge the GeoJSON with your data
    #merged_data = com_data.merge(com_geo, left_on=['CODGEO'], right_on=['code'], how='inner')
    merged_data = com_geo.merge(com_data, left_on=['code'], right_on=['CODGEO'], how='inner')
    clusters = sorted(merged_data['clusters'].unique())  # Sorted unique cluster labels


    category10_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#1f77b4", "#ff7f0e",
    ]

    # Assuming cluster labels from 0 to 9
    #clusters = range(10)
    color_mapping = {str(cluster): category10_colors[i] for i, cluster in enumerate(clusters)}

    m = folium.Map(location=[48.71, 0.71], zoom_start=8, prefer_canvas=True)


    # Function to apply cluster color
    def style_function(feature):
        cluster = feature['properties']['clusters']
        return {
            'fillColor': color_mapping[str(cluster)],
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.6
        }

    highlight_function = lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    
    folium.GeoJson(
        merged_data,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['nom', field],
            aliases=['Commune:', f'{field}:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    # Use branca to generate the legend for clusters
    colormap = cm.StepColormap(
        colors=category10_colors,
        vmin=int(min(clusters)),
        vmax=int(max(clusters)),
        caption="Cluster Colors"
    )

    # Add the colormap legend to the map
    colormap.add_to(m)


    return m


def build_map_clustering_v3(population, model, field, type_ind, n_clusters):
    # m = folium.Map([45.35, -121.6972], zoom_start=12)
# 
    # folium.Marker(
        # location=[45.3288, -121.6625],
        # tooltip="Click me!",
        # popup="Mt. Hood Meadows",
        # icon=folium.Icon(icon="cloud"),
    # ).add_to(m)
   
    df=dict_df_clusters[model]
    ratios_moyens=dict_df_clusters["ratios_moyens"]


    if len(df)==0:
        if model=='PCA': build_pca_clustering(type_ind, population, n_clusters, False)
        if model=='T-SNE': build_tsne_clustering(type_ind, population, n_clusters, False)
        if model=='UMAP': build_umap_clustering(type_ind, population, n_clusters, False)
        df=dict_df_clusters[model]
        ratios_moyens=dict_df_clusters["ratios_moyens"]


    com_data = df[(df["P21_POP"]>=population)]
    
    # recup ZE code + geometrie
    # com_data=com_data.merge(ZE_communes, on="CODGEO", how="left")
    # com_data.drop(columns=['LIBGEO', 'ZE2020_PARTIE_REG', 'DEP', 'REG'], inplace=True)
    # com_data=com_data.merge(ZE_geo, left_on="ZE2020", right_on="ze2020", how="left")
    # com_data.rename(columns={"geometry": "geometry_ZE"}, inplace=True)

    com_data=com_data.merge(np.round(ratios_moyens, 2), left_on='clusters', right_on='Cluster', how='inner')
    # Merge the GeoJSON with your data
    #merged_data = com_data.merge(com_geo, left_on=['CODGEO'], right_on=['code'], how='inner')
    merged_data = com_geo.merge(com_data, left_on=['code'], right_on=['CODGEO'], how='inner')
    #merged_data.rename(columns={"geometry": "geometry_com"}, inplace=True)


    m = folium.Map(location=[48.71, 0.71], zoom_start=8, prefer_canvas=True)

    # Create a colormap
    merged_data["clusters"]=merged_data["clusters"].astype(int)
    
    n_clusters = len(merged_data['clusters'].unique())
    
    
    # Generate blue color palette
    def generate_blue_palette(num_colors):
        return ['#{:02x}{:02x}{:02x}'.format(int(255 * (1 - i / (num_colors - 1))), 
                                            int(255 * (1 - i / (num_colors - 1))), 
                                            255) 
                for i in range(num_colors)]

    blue_colors = generate_blue_palette(n_clusters)
    # Create a colormap
    colormap = cm.LinearColormap(
        colors=blue_colors,
        vmin=merged_data[field].min(),
        vmax=merged_data[field].max()
    )
    colorScheme='Blues'
    
    
    # Add the choropleth layer
    folium.Choropleth(
        geo_data=merged_data,
        name='Communes',
        data=merged_data,
        columns=['CODGEO', field],  
        key_on='feature.properties.code',
        fill_color=colorScheme,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=field  
    ).add_to(m)

    # Add hover functionality without creating a new legend
    style_function = lambda x: {'fillColor': 'transparent', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    folium.GeoJson(
        merged_data,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['nom', "clusters", field],
            aliases=['Commune:', 'Cluster:', f'{field}'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)



    # Add the second layer (ZE_geo) as a larger GeoJSON layer
    folium.GeoJson(
        ZE_geo,
        name="Zones d'emploi",  # Name of the layer in the LayerControl
        style_function=lambda x: {'fillColor': 'blue', 'color': 'red', 'fillOpacity': 0.1, 'weight': 1.5},  # Custom style for ZE layer
        tooltip=folium.features.GeoJsonTooltip(
            fields=['libze2020', 'nb_com'],
            aliases=['Zone:', 'Nb of Communes:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    # Add LayerControl to toggle between layers
    folium.LayerControl().add_to(m)


    # Add color scale
    #colormap.add_to(m)

    legend_css = """
        <style>
        .legend {
            background-color: white;
            border: 2px solid grey;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
        }
        </style>
    """

    # Inject the CSS into the map HTML
    #m.get_root().html.add_child(folium.Element(legend_css))

    return m

# if __name__ == "__main__":
    # main()