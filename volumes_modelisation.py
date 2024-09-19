import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import time
import folium
from branca.colormap import LinearColormap
import streamlit as st
import joblib


root="streamlit"
#root="."

@st.cache_data
def load_data_volumes():
    t=time.time()
    df_periode=pd.read_csv(f"{root}/data/processed/volumes/volumes_quarter.csv.zip", sep=",", dtype={"CODGEO": str, "REG": str, "ZE2020": str}, compression="zip")
    print(f'time data load volumes: {time.time()-t}')    

    return df_periode

@st.cache_data
def load_shapefile_ZE():
    t=time.time()
    ZE_geo = gpd.read_file(f"{root}/data/referentiels/ze2020_2023/ze2020_2023.shp",)    
    print(f'time shapefile ZE volumes: {time.time()-t}')    
    return ZE_geo

@st.cache_data
def get_features(df_periode):
    t=time.time()
    def get_df_lags(df_periode, target, key_feature):
        _agg={
            target+"_annuel": (target, "mean"),
            key_feature+"_annuel": (key_feature, "mean")
        }
        group=df_periode.groupby(['codeCommuneEtablissement','annee']).agg(**_agg).reset_index()

        df_lags=df_periode.merge(group, left_on=['codeCommuneEtablissement','annee'], right_on=['codeCommuneEtablissement','annee'], how="inner")

        
        lags=[
            'lag_5_target', 'lag_6_target', 'lag_7_target','lag_8_target','lag_9_target',
            'lag_10_target', 'lag_11_target','lag_12_target', 'lag_13_target', 'lag_14_target',
        ]


        df_lags['lag_5_target'] = df_lags[target+'_annuel'].shift(20)
        df_lags['lag_6_target'] = df_lags[target+'_annuel'].shift(24)
        df_lags['lag_7_target'] = df_lags[target+'_annuel'].shift(28)
        df_lags['lag_8_target'] = df_lags[target+'_annuel'].shift(32)
        df_lags['lag_9_target'] = df_lags[target+'_annuel'].shift(36)
        df_lags['lag_10_target'] = df_lags[target+'_annuel'].shift(40)
        df_lags['lag_11_target'] = df_lags[target+'_annuel'].shift(44)
        df_lags['lag_12_target'] = df_lags[target+'_annuel'].shift(48)
        df_lags['lag_13_target'] = df_lags[target+'_annuel'].shift(52)
        df_lags['lag_14_target'] = df_lags[target+'_annuel'].shift(56)

        df_lags = df_lags.fillna(0)

        return {"df_lags": df_lags, "lags": lags}
    
        
    df_lags_final={}

    targets=[
        {"step": "creations", 'target': "nb_etabl_crees", "target_pred": "nb_etabl_crees_pred"},
        {"step": "fermetures", 'target': "nb_etabl_fermes", "target_pred": "nb_etabl_fermes_pred"},
    ]

    for target in targets:
        df_lags= get_df_lags(df_periode, target["target"], target['target_pred'])
        df_lags, lags=df_lags["df_lags"], df_lags["lags"]
        df_lags_encoded=pd.get_dummies(data=df_lags, columns=["ZE2020", 'region'])

        df_lags_final[target["step"]]= df_lags_encoded
        df_lags_final['lags']= lags

    encoded_ze=[]
    encoded_region=[]
    for col in df_lags_final["creations"].columns:
        if col.find(("ZE2020"))==0:
            encoded_ze.append(col)
        elif col.find(("region_"))==0:
            encoded_region.append(col)    

    print("time compute features", time.time()-t)
    return df_lags_final, encoded_ze, encoded_region

df_periode = load_data_volumes()
ZE_geo=load_shapefile_ZE()

df_lags_final, encoded_ze, encoded_region=get_features(df_periode)
lags=df_lags_final["lags"]



@st.cache_data
def get_train_test(step, df_lags_final, startYear=2000, cuttingPoint=2018):
    t=time.time()
    def scale_datasets(num_cols, df_train,  df_test, df_valid=None):
        scaler=StandardScaler()
        
        df_train_scaled=scaler.fit_transform(df_train[num_cols])
        df_train_scaled=pd.DataFrame(df_train_scaled, columns=num_cols)
        #df_train_scaled=df_train_scaled.merge(pd.get_dummies(df_train["ZE2020"]), left_index=True, right_index=True)

        df_valid_scaled=None
        if type(df_valid)!=type(None):
            df_valid_scaled=scaler.transform(df_valid[num_cols])
            df_valid_scaled=pd.DataFrame(df_valid_scaled, columns=num_cols)


        df_test_scaled=scaler.transform(df_test[num_cols])
        df_test_scaled=pd.DataFrame(df_test_scaled, columns=num_cols)
        #df_test_scaled=df_test_scaled.merge(pd.get_dummies(df_test["ZE2020"]), left_index=True, right_index=True)
        return {"df_train_scaled": df_train_scaled, 
                "df_test_scaled": df_test_scaled,
                "df_valid_scaled": df_valid_scaled, 
                "scaler": scaler}

    lags=df_lags_final["lags"]
    df_lags_encoded= df_lags_final[step["step"]]
    df_train=df_lags_encoded[(df_lags_encoded["annee"]>= startYear) &(df_lags_encoded["annee"]<= cuttingPoint)].reset_index(drop=True)
    df_test=df_lags_encoded[(df_lags_encoded["annee"]> cuttingPoint) &(df_lags_encoded["annee"]<= cuttingPoint+5)].reset_index(drop=True)


    num_cols=[
        'periode',     
        step["target_pred"],
        'NB_COM',]+lags

    data_scaled= scale_datasets(num_cols, df_train,  df_test,)
    X_train_global, X_test_global, scaler=data_scaled["df_train_scaled"], data_scaled["df_test_scaled"], data_scaled["scaler"]
    X_train_global=pd.merge(X_train_global, df_train[encoded_ze+encoded_region], left_index=True, right_index=True)
    X_test_global=pd.merge(X_test_global, df_test[encoded_ze+encoded_region], left_index=True, right_index=True)
    y_train_global=df_train[step["target"]]
    y_test_global=df_test[step["target"]]

    print("time compute train test", time.time()-t)

    return df_train, df_test, X_train_global, X_test_global, y_train_global, y_test_global, scaler


# get features for all steps (crea & ferm)
df_lags_final, encoded_ze, encoded_region=get_features(df_periode)


# -------------------- prep datasets for pred
# 1. get train-test data X & y for creations
step={"step": "creations", 'target': "nb_etabl_crees", "target_pred": "nb_etabl_crees_pred"}
df_train_creations, df_test_creations, X_train_global_creations, X_test_global_creations, y_train_global_creations, y_test_global_creations, scaler_creations= get_train_test(step, df_lags_final,)

# 2. get train-test data X & y for fermetures
step={"step": "fermetures", 'target': "nb_etabl_fermes", "target_pred": "nb_etabl_fermes_pred"}
df_train_fermetures, df_test_fermetures, X_train_global_fermetures, X_test_global_fermetures, y_train_global_fermetures, y_test_global_fermetures, scaler_fermetures= get_train_test(step, df_lags_final,)


@st.cache_resource
def remake_predict_creations(_models, X_test_global, y_test_global):

    for model in _models:
        # Timing the prediction process
        pred_time = time.time()
        y_pred = model["instance"].predict(X_test_global)    
        model["predict_time"] = time.time() - pred_time

        # Store predictions and metrics
        model["y_pred"] = y_pred
        model["r2_score_test"] = r2_score(y_test_global, y_pred)
        model["rmse"] = root_mean_squared_error(y_true=y_test_global, y_pred=y_pred)
        model["mse"] = mean_squared_error(y_true=y_test_global, y_pred=y_pred)
        model["mae"] = mean_absolute_error(y_true=y_test_global, y_pred=y_pred)
        model["ratio_total"] = y_pred.sum() / y_test_global.sum()        

        print(f'end pred model {[model["name"]]}')

    return _models

@st.cache_resource
def remake_predict_fermetures(_models, X_test_global, y_test_global):

    for model in _models:
        # Timing the prediction process
        pred_time = time.time()
        y_pred = model["instance"].predict(X_test_global)    
        model["predict_time"] = time.time() - pred_time

        # Store predictions and metrics
        model["y_pred"] = y_pred
        model["r2_score_test"] = r2_score(y_test_global, y_pred)
        model["rmse"] = root_mean_squared_error(y_true=y_test_global, y_pred=y_pred)
        model["mse"] = mean_squared_error(y_true=y_test_global, y_pred=y_pred)
        model["mae"] = mean_absolute_error(y_true=y_test_global, y_pred=y_pred)
        model["ratio_total"] = y_pred.sum() / y_test_global.sum()        

        print(f'end pred model {[model["name"]]}')

    return _models

@st.cache_resource
def load_model_creations():
    t=time.time()
    m= joblib.load(filename=f"{root}/models/prediction_volumes_creations.joblib",)
    print("time load model creations", time.time()-t)
    return m

@st.cache_resource
def load_model_fermetures():
    t=time.time()
    m= joblib.load(filename=f"{root}/models/prediction_volumes_fermetures.joblib")
    print("time load model creations", time.time()-t)
    return m

@st.cache_data
def setup_list_models_volumes():
    models={
        "creations_fitted":load_model_creations(),
        "fermetures_fitted": load_model_fermetures()
    }
    return models


def get_stats_by_zone(df_test, model, y_pred, target_type, _target, margin=0.3):
    df_ratio_all_zones=pd.DataFrame([])

    #align global y_pred of the model with y_test that lies in the df_test
    
    ratio_par_zone=[]
        
    for ze in encoded_ze:        
        df_ze=df_test[(df_test[ze]==True)].copy()# & (df_test[target_processing["target_src"]]>0)]        
        y_pred=df_ze["y_pred_"+target_type]
        y_true=df_ze[_target]
        y_pred_sum=y_pred.sum()
        y_true_sum=y_true.sum()
        ratio=y_pred_sum/y_true_sum
        r2=r2_score(y_true, y_pred)
        mape=mean_absolute_percentage_error(y_true, y_pred)
        

        ratio_par_zone.append({ "ratio": ratio, "ze": ze.replace("ZE2020_", ""), "y_real": y_true_sum, "y_pred": y_pred_sum, "r2": r2, "mape": mape, "model": model})


    ratio_global=df_test['y_pred_'+target_type].sum()/df_test[_target].sum()

    df_ratio_par_zone=pd.DataFrame(ratio_par_zone)
    df_ratio_all_zones=pd.concat([df_ratio_all_zones, df_ratio_par_zone], axis=0)

    nb_ze_hors_marge=len(df_ratio_par_zone[(df_ratio_par_zone["ratio"]<1-margin)|(df_ratio_par_zone["ratio"]>1+margin)])
    ratio_hors_ze=np.round(nb_ze_hors_marge/len(df_ratio_par_zone), 2)
    return {
        "df": df_ratio_all_zones,
        "stats":{
            "model": model, 
            "ratio moy par ze": df_ratio_par_zone["ratio"].mean(), 
            "std ratio": df_ratio_par_zone["ratio"].std(), 
            "min": df_ratio_par_zone["ratio"].min(),
            "max": df_ratio_par_zone["ratio"].max(),
            "q25": df_ratio_par_zone["ratio"].describe()["25%"],
            "q75": df_ratio_par_zone["ratio"].describe()["75%"],
            f"nb ze hors marge": nb_ze_hors_marge,
            f"% ze hors marge": ratio_hors_ze
        }
    }


def plot_ratioTotal_by_ZE(top_models, step, sort_alias, margin):
    top_models=np.round(top_models, 2)
    nb_rows=len(top_models["region-ze"].unique())
    color_scheme = px.colors.qualitative.D3  # You can also use px.colors.qualitative.Plotly or Tab10

    fig = px.strip(top_models, 
                x="ratio", 
                y="region-ze", 
                color="model", 
                title=f"Classement des ZE pour les {step.lower()}, trie par {sort_alias.lower()}. Proche de 1 est meilleur.",
                hover_data=["gap_volumes", "y_real", "y_pred"],
                color_discrete_sequence=color_scheme  # Set the color scheme
            )


    # Adding the vertical lines for margin (equivalent to seaborn's axvline)
    fig.add_shape(type="line", x0=1-margin, x1=1-margin, y0=0, y1=nb_rows, line=dict(color="red", dash="dash"))
    fig.add_shape(type="line", x0=1, x1=1, y0=0, y1=nb_rows, line=dict(color="green", width=2))
    fig.add_shape(type="line", x0=1+margin, x1=1+margin, y0=0, y1=nb_rows, line=dict(color="red", dash="dash"))


    # Adjusting the height dynamically based on the number of items in 'top_models'
    fig_height = max(400, nb_rows * 20)  # Adjust based on your data size
    #fig_height = max(400, len(top_models["region-ze"]) * 20)  # Adjust based on your data size
    

    # Updating layout to add grid and other aesthetics
    fig.update_layout(
        height=fig_height,
        xaxis_title="Ratio", 
        yaxis_title="Region-ZE",
        xaxis_showgrid=True, 
        yaxis_showgrid=True, 
        showlegend=True,
        
    )

    return fig


models=setup_list_models_volumes()


def get_scores_global_creations_nettes_volumes(df_test, models):
    y_true=df_test["nb_etabl_crees_net"]
    y_pred=df_test["creations_nettes_pred"]
    mae_crea_nettes=mean_absolute_error(y_true, y_pred)    
    ratio_crea_nettes=y_pred.sum()/y_true.sum()

    all_pred_metrics=[
        {"operation": 'creations_pred', "mae": models["creations_predict"][0]['mae'], "ratio": models["creations_predict"][0]["ratio_total"]},
        {"operation": 'fermetures_pred', "mae": models["fermetures_predict"][0]['mae'], "ratio": models["fermetures_predict"][0]["ratio_total"]},
        {"operation": 'calc_creations_nettes', "mae": mae_crea_nettes, "ratio": ratio_crea_nettes,}
    ]
    return pd.DataFrame(all_pred_metrics)


def get_scores_ZE_creations_nettes_volumes(df_test, df_crea_brutes_by_ZE, df_fermetures_by_ZE, encoded_ze, margin):
    ratio_crea_nettes_par_zone=[]
    for ze in encoded_ze:        
        df_ze=df_test[(df_test[ze]==True)].copy()# & (df_test[target_processing["target_src"]]>0)]        
        y_pred=df_ze["creations_nettes_pred"]
        y_true=df_ze["nb_etabl_crees_net"]
        y_pred_sum=y_pred.sum()
        y_true_sum=y_true.sum()
        ratio=y_pred_sum/y_true_sum
        r2=r2_score(y_true, y_pred)
        mae=mean_absolute_error(y_true, y_pred)
        mape=mean_absolute_percentage_error(y_true, y_pred)
        
        # nb_ze_hors_marge=len(df_ratio_all_zones[(df_ratio_all_zones["ratio"]<1-margin)|(df_ratio_all_zones["ratio"]>1+margin)])
        # ratio_hors_ze=np.round(nb_ze_hors_marge/len(df_ratio_all_zones), 2)
        
        ratio_crea_nettes_par_zone.append({ 
            "ratio": ratio, 
            "ze": ze.replace("ZE2020_", ""),
            "y_real": y_true_sum, 
            "y_pred": y_pred_sum, 
            "mae": mae
        })

    df_ratio_all_zones_crea_net=pd.DataFrame(ratio_crea_nettes_par_zone)    


    nb_ze_hors_marge=len(df_ratio_all_zones_crea_net[(df_ratio_all_zones_crea_net["ratio"]<1-margin)|(df_ratio_all_zones_crea_net["ratio"]>1+margin)])
    ratio_hors_ze=np.round(nb_ze_hors_marge/len(df_ratio_all_zones_crea_net), 2)

    synthese_all_pred=[
        {
            "operation": 'pred_creat._brutes', 
            "ratio_moy/ZE": df_crea_brutes_by_ZE["ratio moy par ze"][0],
            "std_ratio": df_crea_brutes_by_ZE["std ratio"][0],
            "min_ratio": df_crea_brutes_by_ZE["min"][0],
            "max_ratio": df_crea_brutes_by_ZE["max"][0],
            "q25": df_crea_brutes_by_ZE["q25"][0],
            "q75": df_crea_brutes_by_ZE["q75"][0],
            "ze_hors_marge": df_crea_brutes_by_ZE["nb ze hors marge"][0],
            "%_ze_hors_marge": df_crea_brutes_by_ZE["% ze hors marge"][0],
        },
        {
            "operation": 'pred_fermet.', 
            "ratio_moy/ZE": df_fermetures_by_ZE["ratio moy par ze"][0],
            "std_ratio": df_fermetures_by_ZE["std ratio"][0],
            "min_ratio": df_fermetures_by_ZE["min"][0],
            "max_ratio": df_fermetures_by_ZE["max"][0],
            "q25": df_fermetures_by_ZE["q25"][0],
            "q75": df_fermetures_by_ZE["q75"][0],
            "ze_hors_marge": df_fermetures_by_ZE["nb ze hors marge"][0],
            "%_ze_hors_marge": df_fermetures_by_ZE["% ze hors marge"][0],
        },
        {
            "operation": 'calc_creat._nettes', 
            "ratio_moy/ZE": df_ratio_all_zones_crea_net["ratio"].mean(), 
            "std_ratio": df_ratio_all_zones_crea_net["ratio"].std(), 
            "min_ratio": df_ratio_all_zones_crea_net["ratio"].min(),
            "max_ratio": df_ratio_all_zones_crea_net["ratio"].max(),
            "q25": df_ratio_all_zones_crea_net["ratio"].describe()["25%"],
            "q75": df_ratio_all_zones_crea_net["ratio"].describe()["75%"],
            "ze_hors_marge": nb_ze_hors_marge,
            "%_ze_hors_marge": ratio_hors_ze
        }    


    ]

    scores_by_ZE= pd.DataFrame(synthese_all_pred).round(2)
    return scores_by_ZE, df_ratio_all_zones_crea_net


def get_scores_global_volumes(step):    
    display_cols=["name", "mse", "mae","ratio_total","r2_score_train",	"r2_score_test", 'predict_time']
    if step=='Créations brutes':
        models["creations_predict"]=remake_predict_creations(models["creations_fitted"], X_test_global_creations, y_test_global_creations)
        df_test_creations["y_pred_creations"] = models["creations_predict"][0]["y_pred"]
        scores=pd.DataFrame(models["creations_predict"])[display_cols]
        return scores
    elif step=='Fermetures':
        models["fermetures_predict"]=remake_predict_fermetures(models["fermetures_fitted"], X_test_global_fermetures, y_test_global_fermetures)
        df_test_fermetures["y_pred_fermetures"] = models["fermetures_predict"][0]["y_pred"]
        scores=pd.DataFrame(models["fermetures_predict"])[display_cols]
        return scores

    elif step=='Créations b. + fermetures':
        # créa brutes
        models["creations_predict"]=remake_predict_creations(models["creations_fitted"], X_test_global_creations, y_test_global_creations)
        df_test_creations["y_pred_creations"] = models["creations_predict"][0]["y_pred"]
        # fermetures
        models["fermetures_predict"]=remake_predict_fermetures(models["fermetures_fitted"], X_test_global_fermetures, y_test_global_fermetures)
        df_test_fermetures["y_pred_fermetures"] = models["fermetures_predict"][0]["y_pred"]
        # combin scores
        scores_crea=pd.DataFrame(models["creations_predict"])[display_cols]
        scores_ferm=pd.DataFrame(models["fermetures_predict"])[display_cols]

        scores=pd.concat([scores_crea, scores_ferm], axis=0)
        return scores

    elif "Créations nettes":
        # faire predic sur crea brutes et fermetures
        # 1. créa brutes
        models["creations_predict"]=remake_predict_creations(models["creations_fitted"], X_test_global_creations, y_test_global_creations)
        df_test_creations["y_pred_creations"] = models["creations_predict"][0]["y_pred"]
        # 2. fermetures
        models["fermetures_predict"]=remake_predict_fermetures(models["fermetures_fitted"], X_test_global_fermetures, y_test_global_fermetures)
        df_test_fermetures["y_pred_fermetures"] = models["fermetures_predict"][0]["y_pred"]


        y_pred_creations_nettes= calc_creations_nettes(models)
        df_test_creations["creations_nettes_pred"]=y_pred_creations_nettes
        df_test_fermetures["creations_nettes_pred"]=y_pred_creations_nettes
        scores= get_scores_global_creations_nettes_volumes(df_test_creations, models)

        return scores

def get_scores_ZE_volumes(step, sort_key, sort_alias, margin=0.3):
    if step == "Créations brutes":
        target_params=[
            {"target_type": "creations",  "target": "nb_etabl_crees",  "df_test": df_test_creations}
        ]
    elif step == "Fermetures":
        target_params=[
            {"target_type": "fermetures",  "target": "nb_etabl_fermes",  "df_test": df_test_fermetures}
        ]        
    elif step == "Créations nettes":
        target_params=[
            {"target_type": "creations_nettes",  "target": "creations_nettes_pred",  "df_test": df_test_creations}
        ]
    elif step == "Créations b. + fermetures":
        target_params=[
            {"target_type": "creations",  "target": "nb_etabl_crees",  "df_test": df_test_creations},
            {"target_type": "fermetures",  "target": "nb_etabl_fermes",  "df_test": df_test_fermetures}
        ]
        
    

    volumes_by_ZE=[]
    df_ratio_all_zones_volumes=pd.DataFrame([])

    for target in target_params:
        if target["target_type"] == "creations" or target["target_type"]== "fermetures":
            for model in models[f"{target['target_type']}_predict"]:
                y_pred=model["y_pred"]
                data=get_stats_by_zone(target["df_test"], model["name"], y_pred, target["target_type"], target["target"], margin)
                
                stats=data["stats"]
                df_ratios_by_zones=data["df"]
                volumes_by_ZE.append(stats)
                df_ratio_all_zones_volumes=pd.concat([df_ratio_all_zones_volumes, df_ratios_by_zones], axis=0)
            df_volumes_by_ZE= pd.DataFrame(volumes_by_ZE).sort_values(by=f"% ze hors marge")
        elif target["target_type"]=='creations_nettes':
            # stats crea brutes par ZE
            y_pred=models["creations_predict"][0]["y_pred"]
            data=get_stats_by_zone(df_test_creations, "KNN", y_pred, "creations", "nb_etabl_crees", margin)            
            df_crea_brutes_by_ZE= pd.DataFrame([data["stats"]])

            # stats crea brutes par ZE
            y_pred=models["fermetures_predict"][0]["y_pred"]
            data=get_stats_by_zone(df_test_fermetures, "Catboost", y_pred, "fermetures", "nb_etabl_fermes", margin)            
            df_fermetures_by_ZE= pd.DataFrame([data["stats"]])


            df_ratios_by_zones=data["df"]            
            df_ratio_all_zones_volumes=pd.concat([df_ratio_all_zones_volumes, df_ratios_by_zones], axis=0)            

            data_volumes_nets_by_ZE=get_scores_ZE_creations_nettes_volumes(df_test_creations, df_crea_brutes_by_ZE, df_fermetures_by_ZE, encoded_ze, margin)
            df_volumes_by_ZE=data_volumes_nets_by_ZE[0]
            df_ratio_all_zones_volumes=data_volumes_nets_by_ZE[1]
            df_ratio_all_zones_volumes["model"]="Calcul créa. net."

            

    def trunc_libs(m):
        if len(m)>=16:
            return m[:16]+f"..."
        else:
            return m

    df_ref_ze=df_periode[["ZE2020", "LIBZE2020", "region"]].drop_duplicates().reset_index(drop=True)
    top_models=df_ratio_all_zones_volumes.merge(df_ref_ze, left_on='ze', right_on='ZE2020', how="inner")
    top_models["region-ze"]=top_models["LIBZE2020"].apply(trunc_libs)+", "+top_models["region"].str[:4]
    top_models["gap"]= np.abs(top_models["ratio"]-1)
    top_models["gap_volumes"]=top_models["y_pred"]-top_models["y_real"]

    plot= plot_ratioTotal_by_ZE(top_models.sort_values(by=sort_key, ascending=False), step, sort_alias, margin)

    
    return df_volumes_by_ZE, plot, top_models


def calc_creations_nettes(models):
    y_pred_net= models["creations_predict"][0]["y_pred"] - models["fermetures_predict"][0]["y_pred"]
    return y_pred_net


def build_map_ZE_volumes(df, field, alias,):
    #merged_data = com_data.merge(com_geo, left_on=['CODGEO'], right_on=['code'], how='inner')
    keep=['ze2020', 'nb_com', 'geometry', 'ratio', 'ze', 'y_real',
       'y_pred', 'ZE2020',
       'region-ze', 'gap', 'gap_volumes']
    merged_data = ZE_geo.merge(df, left_on=['ze2020'], right_on=['ze'], how='right')[keep]



    if field=="ratio":
        # Custom diverging color map (Green at 1, red below and above)
        custom_colormap = LinearColormap(
            colors=['red', 'orange', 'green','orange', 'red'],  # Red for <1, Green at 1, Yellow for >1
            #vmin=merged_data['ratio'].min(),  # Minimum value of your 'ratio' column
            vmin=0.5,
            vmax=1.5,
            caption='Ratio (vert = meilleur)'
        ).scale(0.5, 1.5)
    elif field=="gap" or field=='gap_volumes':
        custom_colormap = LinearColormap(
            colors=['green', 'yellow', 'orange', 'red'],  # Red for <1, Green at 1, Yellow for >1            
            vmin=merged_data[field].min(),  # Minimum value of your 'ratio' column,
            vmax=merged_data[field].max(),  # Minimum value of your 'ratio' column
            caption='Vert = meilleur'
        ).scale(vmin=merged_data[field].min(), vmax=merged_data[field].max())
    elif field in ["y_real", "y_pred"]:
        #exclure ZE paris pour éviter un gap de couleurs trop important
        merged_data=merged_data[merged_data["ze2020"]!="1109"]
        custom_colormap = LinearColormap(
            colors=["white", 'blue'],  # Red for <1, Green at 1, Yellow for >1            
            vmin=merged_data[field].min(),  # Minimum value of your 'ratio' column,
            vmax=merged_data[field].max(),  # Minimum value of your 'ratio' column
            #caption='Vert = meilleur'
        ).scale(vmin=merged_data[field].min(), vmax=merged_data[field].max())        
        

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
        merged_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['region-ze', field, 'gap', 'gap_volumes'])  # Display additional fields in tooltips
    ).add_to(m)

    # Get the bounds (bounding box) of the GeoDataFrame to fit the map to the geometry
    bounds = merged_data.total_bounds  # [minx, miny, maxx, maxy]

    # Fit the map to the bounds of the geometry (bbox)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Add the custom colormap to the map as a legend
    custom_colormap.add_to(m)


    return m

@st.cache_data
def build_list_ZE_volumes():
    df_ref_ze=df_periode[["ZE2020", "LIBZE2020", "region", 'NB_COM']].drop_duplicates().reset_index(drop=True)
    df_volume_creations_nettes=df_periode[df_periode["annee"]>2018].groupby(["ZE2020"]).agg({"nb_etabl_crees_net": 'sum'}).reset_index()
    df_ref_ze=df_ref_ze.merge(df_volume_creations_nettes, on='ZE2020').sort_values(by="nb_etabl_crees_net", ascending=False)

    def trunc_lib_ZE_volumes(l):
        if l.find('La ')>-1:
            l=l.replace('La ', '')

        if len(l)>=18:
            return l[:18]+f"..."
        else:
            return l    

    df_ref_ze["alias"]= df_ref_ze["LIBZE2020"].apply(trunc_lib_ZE_volumes)+", "+df_ref_ze["region"].str.capitalize()+"     ("+ np.round(df_ref_ze["nb_etabl_crees_net"],0).astype(str)+" créations nettes)"

    return df_ref_ze["alias"], df_ref_ze

def build_curve_volumes(ZE):
    serie_alias, df_ref_ze=build_list_ZE_volumes()
    _ZE=df_ref_ze[df_ref_ze["alias"].isin(ZE)]["LIBZE2020"]
    data=df_test_creations[df_test_creations["LIBZE2020"].isin(_ZE)].reset_index(drop=False)
    
    if 'creations_nettes_pred' not in data.columns:
        return "Vous devez d'abord calculer les créations nettes dans la section précédente"
    
    group=data.groupby(['annee', 'periode', 'quarter', 'LIBZE2020']).agg(
        {"nb_etabl_crees_net": 'sum', 'creations_nettes_pred': 'sum'}
    ).reset_index()
    figs=[]

    for _ze in _ZE:
        _g=group[group["LIBZE2020"]==_ze]
        ratio=np.round(_g["creations_nettes_pred"].sum()/_g["nb_etabl_crees_net"].sum(), 2)
        r2=r2_score(y_pred=_g["creations_nettes_pred"], y_true=_g["nb_etabl_crees_net"])
        mape=np.round(mean_absolute_percentage_error(y_pred=_g["creations_nettes_pred"], y_true=_g["nb_etabl_crees_net"]), 2)
        fig = px.line(_g, x='quarter', y=['nb_etabl_crees_net', 'creations_nettes_pred'],
            labels={'value': 'Number of Creations', 'variable': 'Type'},
            title=f"""Comparaison des volumes nets réels vs predits sur 5 ans pour {_ze}
                <br>
                <sup>Ratio y_pred / y_true: {ratio}</sup>
                <br>
                <sup>MAPE: {mape}</sup>
            """,
        )
        figs.append(fig)
    return figs
# scores=get_scores_global_creations(models_creations)
# print(scores)

