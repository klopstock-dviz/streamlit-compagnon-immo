import streamlit as st
st.set_page_config(layout="wide")
from streamlit_folium import folium_static, st_folium
from data_clustering import get_data_head_clustering, get_data_describe_clustering, get_columns_clustering
from data_clustering import build_hist_population_clustering, build_correlations_clustering, build_pca_clustering, build_tsne_clustering, build_umap_clustering, build_map_clustering_v3
from volumes_modelisation import get_scores_global_volumes, get_scores_ZE_volumes, build_map_ZE_volumes, build_curve_volumes, build_list_ZE_volumes

from dvm_modelisation import get_list_zone_emploi, display_dvm, display_boxplot_ratio, display_dvm_2020, display_lstm_bi_prediction, display_sarima_log, display_metrics_dvm, build_map_zone_emploi, display_model_forecasting, display_boxplot_mape, display_boxplot_rmse


from data_exploration import build_plot_nb_crea_idf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import time
from PIL import Image


#root='streamlit-compagnon-immo'
root='.'

def main():
    t=time.time()
    st.title("Projet compagnon immo")
    st.sidebar.title("Sommaire")
    pages=[
        "Plan de la soutenance", 
        "Introduction de la problèmatique",
        "Pré-processing et exploration", 
        "Prédiction des volumes nets",
        "Prédiction de la durée de vie",
        "Clustering",
        "Conclusion et bilan"]
    
    page=st.sidebar.radio("Aller vers", pages)

    @st.cache_data
    def st_get_list_zone_emploi():
        return get_list_zone_emploi()

    if page == pages[0]:
        st.write("### Plan de la soutenance")
        st.markdown("""
            ### Organisation app streamlit


            * **Page 1**: Introduction de la problèmatique: **2 minutes**
            * **Page 2**: Pré-processing et exploration: **2 minutes**
            * **Page 3**: Prédiction des volumes nets: **5 minutes**
            * **Page 4**: Prédiction de la durée de vie: **5 minutes**
            * **Page 5**: Clustering: **5 minutes**
            * **Page 6**: Conclusion et bilan: **1 minute**
               
            """)
    

    if page == pages[1] : 
        st.markdown("""
        ### Auteurs:<br>
        Ce projet a été développé par <a href='https://www.linkedin.com/in/cruchon-stephane/'>Stéphane CRUCHON</a> et <a href='https://www.linkedin.com/in/aghiles-chougar-04868813/'>Aghiles CHOUGAR</a> 
        entre février et septembre 2024.
        <hr>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### Contexte:
        Dans le cadre de notre formation en science de données chez Datascientest, nous avons travaillé sur un modèle d'aide à la décision de sélection d'emplacement dans une perspective d'achat immobilier.

        Conseiller un acheteur immobilier sur un territoire “pertinent”, “de qualité”, ou “recherché” comme disent les gens du métier, et au delà des inclinaisons culturelles et subjectives que présente l’acheteur, 
        cela revient à conseiller un territoire “dynamique”, sur lequel l’immobilier ne risque pas de se déprécier, et qui maintiendra ou augmentera son attractivité marchande dans un futur proche.

        A première vue tout le monde peut concevoir ce qu’est un territoire dynamique, avec des clichés communs tels que des immeubles qui poussent partout, des emplois disponibles facilement, des infrastructures de qualité, une démographie vigoureuse ...

        Mais afin de produire un indicateur solide et complet, il faut déplier chacun des concepts derrière ces clichés, les relier dans le cadre d’un système organisé, équilibré, et évolutif dans le temps.
        Aller au-delà des idées reçues et des clichés devient compliqué, car nous sommes en effet au croisement de différents savoirs: économie, sociologie, géographie, urbanisme ... 
                    
        ### Objectif du projet:

        L’objectif limité du projet est de proposer des **indicateurs de dynamisme** pour une personne lambda, permettant de comprendre, sur une projection à 5 ans, la tendance associée à la dynamique d’un territoire. 

        L’**économie territoriale** ainsi que la **démographie** ont un impact non négligeable sur le dynamisme.

        #### Focalisation des données

        Sur l’ensemble des données disponibles, nous nous sommes focalisés sur la **base SIREN** qui répertorie un grand nombre d'informations sur les établissements. 

        #### Indicateurs extraits

        Associées aux **zones d’emplois** et par **mois**, nous avons extrait les éléments suivants :

        - La **durée de vie** des établissements
        - Le **nombre de créations**
        - Le **nombre de fermetures** d’établissements
        
        Ces indicateurs "bruts" seront synthétisés sous la forme d'indicateurs métier, plus facilement interprétables:
        > Indicateur sur le volume de création d'établissement:
        > * Rapport entre le cumul du volume de création des 5 années antérieures et le cumul du volume de création des 5 années suivantes
        > Indicateur sur la durée de vie moyenne d'un établissement:
        > * Rapport entre la moyenne de la durée de vie des 5 années antérieures et la moyenne de la durée de vie des 5 années suivantes

        #### Méthodes de modélisation

        Pour modéliser ces variables au cours du temps, nous avons utilisé plusieurs techniques :

        - Des **algorithmes de régression** pour établir des tendances linéaires et non-linéaires.
        - Le modèle **SARIMA** (*Seasonal AutoRegressive Integrated Moving Average*) afin de capturer les effets saisonniers.
        - Des techniques de **Deep Learning** basées sur des réseaux de neurones récurrents (**RNN**), en particulier :
        - Les **LSTM** (*Long Short-Term Memory*), pour gérer les dépendances à long terme dans les séries temporelles.
        - Les **GRU** (*Gated Recurrent Units*), pour un apprentissage efficace avec moins de paramètres tout en maintenant de bonnes performances.

        Ces approches permettent d’obtenir une vision à la fois précise et adaptable du dynamisme d'un territoire. Enfin, nous avons abordé le clustering 
        sur la maille des communes afin d'isoler des sous-ensembles aux caractèristiques communes. 
        """)

    if page == pages[2] : 
        st.write("#### Pré-processing & exploration")
        st.write("---")
        st.markdown("""
            ##### Présentation générale et définitions:
            <br>
            Deux fichiers principaux portant sur l'activité économique sont utilisés:
            <br><br>

                        
            <u>**1. Sirene unités légales**</u>:
            <br>
            Contient la liste des entités juridiques de droit public ou privé. Ces entités juridiques peut être :
            * une personne morale
            * une personne physique
            <br>
            Elle est obligatoirement déclarée aux administrations pour exister.<br>
            Elle est identifiée par le code **siren**.
            
            <br>
                    
            <u>**2. Sirene établissements**</u>: 
            <br>
            Contient la liste des établissements, qui sont des unités de production géographiquement localisées, et juridiquement dépendante de l'unité légale.
            <br>
            Un établissement, qui est identifiée par le code **siret**, produit des biens ou des services, et peut être: 
            * une usine 
            * une boulangerie
            * un magasin de vêtements
            * un hôtels d'une chaîne hôtelière ...
            <br>
            L'établissement, unité de production, constitue le niveau le mieux adapté à une approche géographique de l'économie.
        """, unsafe_allow_html=True)
        st.write("---")
        with open(f"{root}/media/pre-processins ph1.svg", "r") as f:
            svg_pre_processins_ph1 = f.read()

        # Afficher le SVG en tant que HTML
        with st.expander("Flux de traitement des données", expanded=False):

            st.markdown(f'<div style="display: grid; justify-content: center">{svg_pre_processins_ph1}</div>', unsafe_allow_html=True)
        
        st.write("---")
        st.write("Ci-dessous un échantillon du fichier Sirene établissement avec les colonnes utilisées")
        if st.toggle("Voir échantillon", key="sample_sirene"):
            df_sirene_sample_explor=pd.read_csv(f"{root}/data/processed/sirene/sirene_head.csv", dtype={'codeCommuneEtablissement':str, 'siren': str, "siret": str, "trancheEffectifsEtablissement": str},)
            st.dataframe(df_sirene_sample_explor.head())
            st.write("---")
            st.dataframe(df_sirene_sample_explor.head(1).T)
        
        st.write("---")
        
        st.write("Ci-dessous un aperçu des valeurs manquantes")
        if st.toggle("Voir NaN", key="NaN_sirene_explor"):
            st.write('% de NaN sur tout le dataset')
            NaN_1_sirene_explor=Image.open(f"{root}/media/sirene_nan.png", "r")
            st.image(NaN_1_sirene_explor, use_column_width=False,)

            st.write("---")
            st.write('% de NaN sur les tranches effectifs en IDF')
            NaN_2_sirene_explor=Image.open(f"{root}/media/sirene_trancheE_nan_idf.png", "r")
            st.image(NaN_2_sirene_explor, use_column_width=False,width=600)


            st.write("---")
            st.write('Compositon des tranches effectifs connues en IDF')
            NaN_3_sirene_explor=Image.open(f"{root}/media/trancheE_connues_IDF.png", "r")
            st.image(NaN_3_sirene_explor, use_column_width=False,width=600)   

        st.write("Ci-dessous des graphiques d'exploration")
        if st.toggle("Voir les graphiques", key="plots_explor"):
            plot_nb_crea_idf = build_plot_nb_crea_idf()
            for el in plot_nb_crea_idf:
                st.plotly_chart(el)

        st.write("Ci-dessous l'extraction de la durée de vie moyenne")
        if st.toggle("Voir les graphiques", key="plots_dvm"):
            
            options_dvm = st.multiselect("Choississez vos zones d'emploi",
                                st_get_list_zone_emploi(),
                                ["Caen", "Rouen"],
                                )   

            fig = display_dvm(options_dvm)
            st.plotly_chart(fig, use_container_width=True) 



    #"Prédiction des volumes nets"
    if page == pages[3]:

        # Initialize session state to store snapshots if it doesn't exist yet
        if "snapshots" not in st.session_state:
            st.session_state["snapshots"] = []

        
        st.write("### Objectif:")
        st.markdown("""
            L'une des mesures possibles du dynamise des territoires est le volume net de création d'établissements sur une période donnée.
            <br>
            Nous avons retenu un horizon de 5 ans, qui est:
            > * Un minimum pour un achat immobilier
            > * Une zone de sécurité pour éviter des erreurs de prédiction importantes
            
            <br>
                    
            **Propriétés de la cible:**
            > * S'obient en calculant l'écart entre créations brutes et fermetures d'établissements
            > * Requiert la prédiction des créations brutes et des fermetures séparement
            > * Le résultat élémentaire de la prédiction est une quantité, par trimestre et par commune
            
            <hr>

            **Projection de la cible:**<br>""", 
        unsafe_allow_html=True)
        detail_eval_modeles_volumes=st.toggle('Voir explications', key="detail_eval_modeles_volumes")
        if detail_eval_modeles_volumes:
            st.markdown("""
                <u>**Projection de la cible pour l'évaluation des modèles:**</u><br>
                > Les volumes de créations / fermetures présentés ci-dessous ont été prédits sur l'horizon 2018-2023.<br>
                > Cela correspond à une pratique habituelle dans la prédiction des séries temporelles, où on cache au modèle une partie du
                > futur connu, en demandant à ce dernier de tenter de le prédire.<br>
                > En procédant de la sorte nous pouvons ainsi évaluer la perfomance du modèle en calculant l'écart en le chiffre réel et sa prédiction.
                > <br>
                > Comme nous avons besoin d'une valeur normalisée pour comparer les performances entre chaque zone d'emploi, nous avons retenu le ratio
                > volumes prédits par le modèle / volumes réels, un ratio idéal étant de 1 (volume prédit = volume réel)
                
                <br><br>
                        
                <u>**Projection de la cible pour un usage pratique:**</u><br>
                > Un indicateur concret et utile serait le ratio volumes prédits (2025-2030) / volumes passées (2020-2024).
                > <br>
                > <br>
                > Si nous nous intéressons aux volumes de créations nettes d'établissements, nous pourrons dire que:
                > * Un ratio proche de 1 signifie que le volume d'établissements prédit sur 5 ans est proche du passé -> zone d'emploi stable.
                > * Un ratio inférieur à 1 signifie que les créations nettes sur les 5 prochaines années sera plus faible que sur les 5 dernières -> zone d'emploi atone.
                > * Un ratio supérieur à 1 signifie que les créations nettes sur les 5 prochaines années sera plus élevé que sur les 5 dernières -> zone d'emploi dynamique.
                > <br>
                > 
                > De cette façon nous pouvons calculer un indicateur générique et facile à comprendre.<br>
                > Malheureusement par manque de temps il n'a pas été possible de ré-entrainer les modèles sur cet horizon de prédiction cible, 
                > cet indicateur métier n'est donc pas exposé pour cette soutenance.
                
            """, unsafe_allow_html=True)
        st.write("---")
        st.write("### Etapes:")
        # Lire le fichier diag de flux
        with open(f"{root}/media/prep data prediction.drawio.svg", "r") as f:
            svg_content = f.read()

        # Afficher le SVG en tant que HTML
        with st.expander("Flux de traitement des données", expanded=False):
            st.markdown(f'<div style="display: grid; justify-content: center">{svg_content}</div>', unsafe_allow_html=True)
            st.write("---")

        with st.expander("Expérimentations et modèles", expanded=False):
            st.markdown("""
                **Premiers essais:**
                1. Modèles de régression linaire par zones d'emploi, aux mailles suivantes:
                > * Commune et année
                > * Commune et mois
                <br>
                2. Modèle unique par zone d'emploi, aux mailles suivantes:
                > * Zone d'emploi et année
                > * Zone d'emploi et mois
                > * Zone d'emploi et trimestre
                <br>
                3. Modèle unique par commune, au trimestre
                4. Introduction de lags temporels, sur 5 ans puis 10 ans
                5. Standard scaling des features
                6. Introduction en feature du volume global par région prédit sur 5 ans (Sarima)
            """, unsafe_allow_html=True)
            if st.toggle("Montrer courbe", key='pred_sarima_volumes'):
                pred_sarima_volumes = Image.open(f"{root}/media/pred_sarima_volumes.png", "r")
                st.image(pred_sarima_volumes, use_column_width=False, width=800)
            
            st.write("Le dataset obtenu en 6 a servi de base à l'entrainement des modèles ci-dessous:")
            if st.toggle("Montrer modèles", key='show_models_pred_volumes'):
                st.markdown("""
                    <u>**Les modèles de régression linéraire:**</u>
                    > * LinearRegression avec PolynomialFeatures (ordre 2)
                    > * ElasticNet
                    > * LinearSVR
                    > * RidgeCV
                    > * LinearRegression
                    
                    <u>**Modèles non linéaires:**</u>
                    > * KNeighborsRegressor
                    > * Catboost
                    > * XGBoost                    
                    > * SVR (kernel=rbf)     
                    > * RandomForestRegressor               
                    > * LightGBM
                    > * DecisionTreeRegressor                    
                    > * GradientBoostingRegressor
                    > * HistGradientBoostingRegressor
                    > * VotingRegressor                    
                    
                    <u>**Réseau de neurones:**</u>
                    > * LSTM
                    > * Réseau dense                    
                            
                """, unsafe_allow_html=True)


        #st.cache_data
        def st_get_scores_ZE_volumes():
            scores_locaux_volumes, plot_scores_ze_volumes, df_ratios= get_scores_ZE_volumes(select_type_pred_volumes, sort_by_field_ZE_volumes, sort_by_alias_ZE_volumes, margin_tolerance_volumes,)
            return scores_locaux_volumes, plot_scores_ze_volumes, df_ratios

        tab1, tab2, tab3 = st.tabs(["Scores", "Carte", "Courbes volumes réels vs prédits"])
        with tab1:
            col1_volumes_params, col2_volumes_params=st.columns(2)
            with col1_volumes_params:
                select_type_pred_volumes= st.selectbox("Type de prédiction", ["Créations brutes", "Fermetures", "Créations nettes", "Créations b. + fermetures"], index=0)
                ploting_fields_volumes= ["Ratio (y_pred/y_true)", "Ecart-type", "Ecart volumes", "Volumes réels", "Volumes prédits"]
                sort_by_alias_ZE_volumes = st.selectbox(
                    "Trier du classement des ZE:",
                    options=ploting_fields_volumes,
                    index=0  # Default sort by ratio
                )
                dict_sortBy_volumes={
                    "Ecart-type": 'gap', 
                    "Ratio (y_pred/y_true)": 'ratio',
                    "Ecart volumes": "gap_volumes",
                    "Volumes réels": "y_real", 
                    "Volumes prédits": "y_pred"
                }
                sort_by_field_ZE_volumes=dict_sortBy_volumes[sort_by_alias_ZE_volumes]





            with col2_volumes_params:
                margin_tolerance_volumes= st.slider("Affiner la marge de tolérance des zones d'emploi", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
            calc_volumes=st.button("Calculer", key="calc_volumes")


            if calc_volumes:
                st.write("###### Scores globaux:")
                scores_volumes_globaux= get_scores_global_volumes(select_type_pred_volumes)
                st.dataframe(scores_volumes_globaux)

                st.write("---")

                st.write("###### Scores locaux par zones d'emploi:")
                scores_locaux_volumes, plot_scores_ze_volumes, df_ratios=st_get_scores_ZE_volumes()
                #scores_locaux_volumes, plot_scores_ze_volumes, df_ratios= get_scores_ZE_volumes(select_type_pred_volumes, sort_by_field_ZE_volumes, sort_by_alias_ZE_volumes, margin_tolerance_volumes,)
                st.dataframe(scores_locaux_volumes)

                st.plotly_chart(plot_scores_ze_volumes)

                #if st.button("Take Snapshot"):
                # Store the current plot in session state (as a dictionary)
                st.session_state["snapshots"].append({
                    "plot": plot_scores_ze_volumes,
                    "title": f"Snapshot {len(st.session_state['snapshots']) + 1}"
                })            

                


                # If snapshots exist, display them
                if st.session_state["snapshots"]:

                    with st.popover("Open snaps", use_container_width=True):
                        #st.markdown("Comparaison de gr")
                        plot=st.session_state["snapshots"][0]["plot"]
                        col1_volumes_snap, col2_volumes_snap=st.columns(2, )
                        with col1_volumes_snap:
                            st.write('Graphique précédent')
                            st.plotly_chart(plot)
                        with col2_volumes_snap:
                            st.write('Graphique actuel')
                            st.plotly_chart(plot_scores_ze_volumes)
        with tab2:            
            ind_alias_map_volumes= st.selectbox("Variable à représenter sur la carte",
                ploting_fields_volumes, 
            index=0)

            ind_field_map_volumes=dict_sortBy_volumes[ind_alias_map_volumes]
        
            btn_calc_map_volumes= st.button("Calculer", key="btn_calc_map_volumes")
            if btn_calc_map_volumes:
                get_scores_global_volumes("Créations nettes",)
                scores_locaux_volumes, plot_scores_ze_volumes, df_ratios=st_get_scores_ZE_volumes()


                if select_type_pred_volumes!="Créations b. + fermetures":                    
                    #["Ratio (y_pred/y_true)", "Ecart-type", "Ecart volumes", "y_real", "y_pred"]
                    map_volumes=build_map_ZE_volumes(df_ratios, ind_field_map_volumes, ind_alias_map_volumes)
                    st.markdown(f"{ind_alias_map_volumes} portant sur les {select_type_pred_volumes.lower()}")              
                    folium_static(map_volumes, width=1200, height=700)
                elif select_type_pred_volumes=="Créations b. + fermetures":
                    st.markdown(f"""<span style='color: salmon'>
                        Il n y a pas de carte disponible pour les {select_type_pred_volumes}""",
                        unsafe_allow_html=True
                    )

        with tab3:
            serie_multiSelect_ZE_volumes, df_ref_ze_volumes= build_list_ZE_volumes()
            ze_courbe_volumes= st.multiselect("Choississez vos zones d'emploi", serie_multiSelect_ZE_volumes,)
            
            btn_courbe_volumes=st.button("Calculer", key='btn_courbe_volumes')
            if btn_courbe_volumes:
                get_scores_global_volumes("Créations nettes",)
                curve_plot_volumes= build_curve_volumes(ze_courbe_volumes)

                if type(curve_plot_volumes)==str:
                    st.markdown(f"<span style='color: salmon'>{curve_plot_volumes}</span>", unsafe_allow_html=True)
                else:
                    for plot_ze_volumes in curve_plot_volumes:
                        st.plotly_chart(plot_ze_volumes)
                
                #df_courbe_volumes=df_ratios[df_ratios["LIBZE2020"]==ze_courbe_volumes]

    if page == pages[4] : 
        st.write("Prédictions de la durée de vie moyenne d'un établissement")
        st.write("#### Objectif:")
        multi_dvm = '''&nbsp;&nbsp;&nbsp;&nbsp;La prédiction de la durée de vie moyenne d'un établissement repose sur l'extraction de données
                    de la base **SIREN**. La maille utilisée est la zone d'emploi comme l'a défini l'INSEE.
                    La profondeur de données disponibles permet d'aller très loin ( années 60's ). Nous avons resserré la fenêtre d'observation afin d'éliminer l'effet 
                    de collecte des données ( bas de données SIREN peuplées à des dates non périodiques, etc ... ). L'objectif est de retenir le meilleur modèle afin de l'utiliser 
                    pour calculer notre indicateur final sur un horizon de 5 ans à partir de la date actuelle. Ces données se présentant
                    sous la forme de série temporelle, il a été décidé d'aborder en premier lieu plusieurs techniques pour les prédictions.
                    
                    Les algorithmes suivants ont été expérimentés et évalués :
                '''
        st.markdown(multi_dvm)
        st.markdown("""
        - Modèle SARIMA(X) avec différents _Feature Engineering_ 
        - Modèles _Deep Learning_ comme LSTM et GRU et ajout de _Feature_
                    """)
        
        st.write("""&nbsp;&nbsp;&nbsp;&nbsp;Nous avons décidé suite au rapport final de ne garder que les modèles de chaque type présentant un ratio moyen optimal à savoir : """)
        st.markdown("""
        >Le modèle **SARIMA avec transformation log**  
        >Le modèle **LSTM bidirectionnel**
                    """)
        # prep data
        @st.cache_data
        def st_display_lstm_bi_prediction(zone_emploi_dvm, n_forecast, sequence_type):
            return display_lstm_bi_prediction(zone_emploi_dvm, n_forecast, sequence_type)
        
        @st.cache_data
        def st_display_sarima_log(zone_emploi_dvm, n_forecast, type):
            return display_sarima_log(zone_emploi_dvm, n_forecast, type)
                
        @st.cache_data
        def  st_display_dvm_2020(pow) :
            return  display_dvm_2020(pow)
        
        @st.cache_data
        def  st_display_metrics_dvm() :
            return  display_metrics_dvm()

        @st.cache_data
        def  st_display_boxplot_ratio() :
            return  display_boxplot_ratio()
        
        @st.cache_data
        def  st_display_boxplot_mape() :
            return  display_boxplot_mape()
        
        @st.cache_data
        def  st_display_boxplot_rmse() :
            return  display_boxplot_rmse()
        
                
        @st.cache_data
        def  st_display_model_forecasting(zone_dvm, forecast) :
            return  display_model_forecasting(zone_dvm, forecast)        
        

        st.markdown(" #### Comparaison de l'évolution de la durée de vie suivant les zones d'emploi")
        st.markdown("""&nbsp;&nbsp;&nbsp;&nbsp;Le critère d'évaluation principal de nos modèles est le **ratio moyen** entre les prédictions et les valeurs réelles.
                    La fenêtre étant de 5 ans, le cutting point de nos séries temporelles à été établi à 2018. Les autres _métriques_ utilisées sont la MAPE et la RMSE.
                    """)
        
        st.markdown(" #### Evaluation des différents modèles et prédictions")
        with st.expander("Flux de traitement des données", expanded=False):
            st.image(f"{root}/media/dvm_engineering.png")   
        with st.expander("Performances des deux modèles retenus : Sarima (log) et LSTM bidirectionnel ", expanded=False):
            line_tab_dvm, point_tab_dvm, point_mape_tab_dvm, point_rmse_tab_dvm, map_tab_dvm = st.tabs(["Comparaison des métriques ", "Distribution du ratio moyen", "Distribution de la MAPE", "Distribution de la RMSE" , "Carte ratio par zone d'emploi"])

            with line_tab_dvm:
                st.plotly_chart(st_display_metrics_dvm(), use_container_width=True)
            with point_tab_dvm:
                st.plotly_chart(st_display_boxplot_ratio(), use_container_width=True)
            with point_mape_tab_dvm:
                st.plotly_chart(st_display_boxplot_mape(), use_container_width=True)
            with point_rmse_tab_dvm:
                st.plotly_chart(st_display_boxplot_rmse(), use_container_width=True)

            with map_tab_dvm:
            
                modele_carte_dvm = st.radio("Choississez le modèle pour l'affichage du ratio :",
                                        ["Sarima log", "LSTM bidirectionnel"], 
                                        horizontal=True, key='modele_carte_dvm'
                                        )
                if modele_carte_dvm == "Sarima log" :
                    field_ratio_dvm = 'sarima_ratio'
                else :
                    field_ratio_dvm = 'lstm_bi_ratio'            

                map_ratio_dvm = build_map_zone_emploi(field_ratio_dvm)
                folium_static(map_ratio_dvm, width=1000, height=700)
        
        with st.expander("Modèlisation SARIMA (log) ", expanded=False):
            st.write("Evaluation et affichage map")

            zone_emploi_sarima_dvm = st.selectbox("Choisir une zone d'emploi ",
                                    st_get_list_zone_emploi(), key="zone_emploi_sarima_dvm"
                                    )      

            if zone_emploi_sarima_dvm :
      
                fig_sarima_dvm = display_sarima_log(zone_emploi_sarima_dvm, 60, 'prediction_log')
                st.plotly_chart(fig_sarima_dvm, use_container_width=True)

            
        with st.expander("Modèlisation de type réseaux de neurones récurrents LSTM (bidirectionnel) ", expanded=False):

            st.write("Evaluation et affichage map ")
            # TODO : Afficher une box rétractable avec comme label "Détails"
            # Expliquer le modèle choisi ainsi que la cross validation avec enregistrement des meilleurs paramètres -> sauvegarde des modèles
            # Appel en live des modèles pour la prédiction 
            
            zone_selection_dvm, horizon_prediction_col_dvm = st.columns(2)
            
            with zone_selection_dvm:
                zone_emploi_rnn_dvm = st.selectbox("Choisir une zone d'emploi ",
                                    st_get_list_zone_emploi(), key="zone_emploi_rnn_dvm"
                                    )        

                sequence_prediction_dvm = st.toggle("off : sequence 5 / on : sequence 12 ", key='sequence_prediction_dvm')

                if sequence_prediction_dvm :
                    sequence_type = 'highseq'
                else :
                    sequence_type = 'lowseq'      
            with horizon_prediction_col_dvm:             
                horizon_step_dvm = st.slider('Horizon de la prédiction ( en mois )', min_value=12, max_value=60, step=12, value=60)

         
            if zone_emploi_rnn_dvm :
      
                fig_rnn_dvm = st_display_lstm_bi_prediction(zone_emploi_rnn_dvm, n_forecast=horizon_step_dvm, sequence_type=sequence_type)
                st.plotly_chart(fig_rnn_dvm, use_container_width=True)
        
        st.markdown(" #### Forecasting sur 5 ans de la durée de vie moyenne")
        with st.expander("Graphe comparatif et répartition géographique", expanded=False):
            st.write("Affichage du __forecast__ généré par les modèles Sarima (log) et LSTM (birectionnel)")
            st.write("Les données prédites au delà des observations connues vont être utilisées pour calculer notre indicateur métier.")
            forecast_tab_dvm, forecast_map_tab_dvm = st.tabs(["Forecast", "Répartition géographique de l'indicateur"])

            with forecast_tab_dvm:
                zone_emploi_forecast_dvm = st.selectbox("Choisir une zone d'emploi ",
                                    st_get_list_zone_emploi(), key="zone_emploi_forecast_dvm"
                                    )  
                if zone_emploi_forecast_dvm :      
                    fig_forecast_dvm = st_display_model_forecasting(zone_emploi_forecast_dvm, 60)
                    st.plotly_chart(fig_forecast_dvm, use_container_width=True)
            
            with forecast_map_tab_dvm:  
                st.markdown("""Répartition de l'indicateur métier par zone d'emploi  ( Année pivot 2023 - indicateur = Moyenne entre 2024-2028 / Moyenne entre 2019-2023 )""") 
                
                with st.container():
                    detail_forecast_map_tab_dvm=st.toggle('Voir explications', key="detail_forecast_map_tab_dvm")
                    if detail_forecast_map_tab_dvm:            
                        st.write("""
                                - Si la valeur de l'indicateur est > 1 alors la durée de vie moyenne connait une évolution positive sur une maille à 5 ans  
                                - Si la valeur de l'indicateur est égale ou très proche de 1 alors la durée de vie moyenne connait une stabilisation sur une maille à 5 ans  
                                - Si la valeur de l'indicateur est < 1 alors la durée de vie moyenne des établissements connait une baisse sur une maille à 5 ans  
                                """)                    

                forecast_carte_dvm = st.radio("Choississez le modèle pour l'affichage de l'indicateur :",
                                            ["Sarima log", "LSTM bidirectionnel"], 
                                            horizontal=True, key='forecast_carte_dvm'
                                            )
                
                if forecast_carte_dvm == "Sarima log" :
                    field_indicateur_dvm = 'indicateur_forecast_sarima'
                else :
                    field_indicateur_dvm = 'indicateur_forecast'            

                map_forecast_dvm = build_map_zone_emploi(field_indicateur_dvm)
                folium_static(map_forecast_dvm,  width=1000, height=700)
            
     

    if page==pages[5]:
        st.write("### Clustering")

        st.write("#### Objectif:")
        st.markdown("""
            N'ayant pas pu exploiter les autres dimensions du dynamisme des territoires, faute de series temporelles longues, nous avons
            voulu dans cette section représenter ces dimensions avec des modèles de clustering, avec pour objectif de:
            * Détecter automatiquement des groupes de communes similaires
            * Vérifier si nos indicateurs de dynamisme issus de nos prédictions sont corrélés aux autres dimensions représentées ici
        """, unsafe_allow_html=False)

        # prep data
        @st.cache_data
        def st_data_head_clustering():
            return get_data_head_clustering()

        # prep data
        @st.cache_data
        def st_data_desc_clustering():
            return get_data_describe_clustering()

        @st.cache_data
        def st_columns_clustering():
            return get_columns_clustering()


        @st.cache_data
        def st_hist_population_clustering():
            return build_hist_population_clustering()

        @st.cache_data
        def st_correlations_clustering():
            return build_correlations_clustering()




        with st.expander("##### Afficher les informations sur le dataframe", expanded=False):
            # gestion affichage tableau head
            st.write("###### Afficher le top 5 du tableau")
            show_df_head_clustering = st.radio("Head", ["Masquer", "Afficher"], horizontal=True)

            
            if show_df_head_clustering == "Afficher":
                df_clustering = st_data_head_clustering()
                st.dataframe(df_clustering)
            # else:
            #     st.write("#### Sélectionnez 'Afficher' pour voir le tableau.")

            st.html('<hr style="margin: 5px">')

            # gestion affichage tableau describe
            st.write('###### Afficher la description du tableau')
            show_df_desc_clustering = st.radio("Describe", ["Masquer", "Afficher"], horizontal=True)
            if show_df_desc_clustering == "Afficher":
                df_clustering = st_data_desc_clustering()
                st.dataframe(df_clustering)
            # else:
            #     st.write("Sélectionnez 'Afficher' pour voir la description.")

            st.html('<hr style="margin: 5px">')

            # gestion affichage tableau variables clustering
            # Create radio buttons
            st.write('###### Afficher les variables')
            show_variables_clustering = st.radio("Variables", ["Masquer", "Afficher"], horizontal=True)

            # Display or hide the dataframe based on the radio button selection
            if show_variables_clustering == "Afficher":
                df_clustering = st_columns_clustering()
                st.dataframe(df_clustering)

            st.html('<hr style="margin: 5px">')

            st.write('###### Afficher les corrélations')
            show_correlations_clustering = st.radio("Corrélations", ["Masquer", "Afficher"], horizontal=True)            
            thresh_corr_clustering=st.slider('Choisir le seuil de sensibilté', min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            # Display or hide the dataframe based on the radio button selection
            if show_correlations_clustering == "Afficher":                
                if st.button('Calculer', key='calc_corr_clustering'):
                    df_correlations_clustering = build_correlations_clustering(thresh_corr_clustering)
                    st.plotly_chart(df_correlations_clustering)


        with st.expander("##### Choisir les paramètres du clustering", expanded=False):
            
            # param 1; seuil de population
            st.markdown("<span style='text-decoration: underline;'>1. Seuil de population par communes</span>", unsafe_allow_html=True)
            num_habitants_clustering=st.slider('Communes ayant au moins n habitants', min_value=1000, max_value=20000, step=1000, value=1000)#, 3000, 5000, 10000)
            
            # graph répartition nb communes / tranches population
            show_population_distribution_clustering = st.radio("Afficher le nombre de communes selon les tranches d'habitants", ["Masquer", "Afficher"], horizontal=True)
            if show_population_distribution_clustering == "Afficher":
                population_distribution=st_hist_population_clustering()
                population_distribution= pd.DataFrame(population_distribution).reset_index(drop=True)
                
                # plot
                st.bar_chart(data=population_distribution, x="population_category", y="count", x_label="Tranches de population", y_label='Nombre de communes')

                st.write("""
                    La marorité des communes sont dans les tranches 0-1000 habitants.\
                    Ces communes ne connaissant presque aucune activité économique en dehors de rares petits commerces.\
                    Nous avons décidé de les exclure du clustering.
                """)
            
            st.html('<hr style="margin: 5px">')

            # param 2: méthode de partitionnement
            st.markdown("""<span style='text-decoration: underline;'>
                2. Modèle de réduction de dimention</span>"""
            , unsafe_allow_html=True)

            dim_red_model_clustering = st.selectbox(
                "Choisir un modèle de réduction de dimension",
                #("PCA", "Isomap", "LLE", 'T-SNE', 'UMAP'), index=4
                ("PCA", 'T-SNE', 'UMAP'), index=0
            )
            
            st.write("Vous avez choisi:", dim_red_model_clustering)
            

            detail_red_dim_clustering=st.toggle('Voir explications', key="detail_red_dim")
            if detail_red_dim_clustering:            
                st.write("""Notre tableau contient un nombre considérable de colonnes.
                        Afin de fournir une représentation en 2d de la répartition des communes en groupes homogènes, nous allons procéder à une réduction de dimension
                """)            

            st.html('<hr style="margin: 5px">')

            # param 3: ensemble d'indicateurs
            st.markdown("""<span style='text-decoration: underline;'>
                3. Nature des indicateurs à traiter</span>"""
            , unsafe_allow_html=True)

            type_ind_clustering = st.selectbox(
                "Choisir un type d'indicateurs à traiter",
                ("Société", "Immobilier", "Economie", "Tout"), index=3
            )

            st.write("Vous avez choisi:", type_ind_clustering)
            detail_indicateurs_clustering=st.toggle('Voir explications', key="detail_indicateurs")
            if detail_indicateurs_clustering:
                st.write("""Notre tableau contient des indicateurs de divers types.
                        Afin de fournir une représentation de la répartition des communes selon une dimension particulière, 
                        vous pouvez choisir celle ci dans la liste ci-dessus.
                """)

            st.html('<hr style="margin: 5px">')

            # param 4: nb de clusters
            st.markdown("""<span style='text-decoration: underline;'>
                4. Nombre de clusters</span>""", unsafe_allow_html=True)
            
            num_clusters_clustering=st.slider('Choisir le nombre de clusters', min_value=6, max_value=12, step=1, value=8)#, 3000, 5000, 10000)

            detail_clustering=st.toggle('Voir explications', key="detail_clustering")
            if detail_clustering:
                st.markdown("""Ici nous ne proposons que le modèle Kmeans, qui offre le résultat le plus satisfaisant pour notre problème.
                    Nous utilisons la variante légère Mini batch Kmeans qui est plus rapide et presque aussi précise que Kmeans.
                         <br>
                    Il est d'usage d'utiliser la méthode du coude et le score de silhouette afin de déterminer automatiquement le nombre optimal
                    de clusters.<br> 
                    Cepdendant le nombre obtenu de cette façon (3) ne capture pas de façon assez fine les différentes zones proposées par 
                    les modèles de réduction de dimensions.<br>
                    Ainsi nous proposons par défaut le nombre de 8, qui semble un bon compromis pour la majorité des réductions calculées. 
                """, unsafe_allow_html=True)

        
            st.html('<hr style="margin: 5px">')

            display_interactive_scatter_clustering=st.toggle('Nuage de points interactif', key="display_interactive_scatter")


            models_red_dim={
                "PCA": build_pca_clustering,
                "T-SNE": build_tsne_clustering,
                "UMAP": build_umap_clustering,
            }

            # param 4: nb de clusters
            st.markdown("""<span style='text-decoration: underline;'>
                5.Paramètres de la carte</span>""", unsafe_allow_html=True)
            
            l_clustering=st_columns_clustering()
            if type_ind_clustering!="Tout":
                l_clustering=l_clustering[(l_clustering["dimension"]==type_ind_clustering)&(l_clustering["active"])][["field", "alias"]].reset_index(drop=True)
                _inds_alias_map_clustering=l_clustering['alias']
            else:
                _inds_alias_map_clustering=l_clustering['alias']
            _inds_alias_map_clustering=["Ratios moyens"]+ list(_inds_alias_map_clustering.head(8))                

            #_inds_map_list=("Ratios moyens", "P21_POP", "MED21")
            ind_map_clustering = st.selectbox(
                "Choisir un indicateur à afficher",
                _inds_alias_map_clustering, 
                index=0,
            )
            print('update ind selectbox ')
            
            ind_map_alias_clustering=ind_map_clustering
            if ind_map_clustering!="Ratios moyens":                
                ind_map_clustering=l_clustering[l_clustering['alias']==ind_map_clustering]["field"].values[0]
                
 
        btn_calc_clustering= st.button('Calculer', type='primary')

        if btn_calc_clustering:
            @st.cache_data
            def get_plots_clustering(dim_red_model, type_ind, num_habitants, num_clusters, display_interactive_scatter):
                print("refresh plots")
                plots= models_red_dim[dim_red_model](type_ind, num_habitants, num_clusters, display_interactive_scatter)
                return plots

            @st.cache_data
            def display_map_clustering(dim_red_model, num_habitants, ind_map_clustering, type_ind, num_clusters):
                print("refresh map")
                map_clustering= build_map_clustering_v3(model=dim_red_model, population=num_habitants, field= ind_map_clustering, type_ind=type_ind, n_clusters=num_clusters)
                
                return map_clustering
            
            plots=get_plots_clustering(dim_red_model_clustering, type_ind_clustering, num_habitants_clustering, num_clusters_clustering, not display_interactive_scatter_clustering)
            scatter=plots["scatter"]
            heatmap=plots["heatmap"]            
            


        tab1_clustering, tab2_clustering, tab3_clustering = st.tabs(["Nuage de points", "Heatmap", "Carte"])

        with tab1_clustering:
            if btn_calc_clustering:
                st.altair_chart(scatter)#, use_container_width=True)

        with tab2_clustering:
            if btn_calc_clustering:
                st.altair_chart(heatmap)

        with tab3_clustering:
            if btn_calc_clustering:        
                st.markdown("""
                <style>.stfolium-container {width: 100%; height: 100vh; }</style>""", unsafe_allow_html=True)
                st.write(f"###### Communes selon l'indicateur {ind_map_alias_clustering}")
                #map_clustering= display_map_clustering(dim_red_model_clustering, num_habitants_clustering, , type_ind_clustering, num_clusters_clustering)
    
                map_clustering= build_map_clustering_v3(model=dim_red_model_clustering, population=num_habitants_clustering, field= ind_map_clustering, type_ind=type_ind_clustering, n_clusters=num_clusters_clustering)
                folium_static(map_clustering, width=1000, height=700)

        #st.write(f"reload time: {time.time()-t}")
        print(f"reload time clustering: {time.time()-t}")

    if page==pages[6]:
        st.markdown("""
            ## Conclusions et perspective:

            ### Conclusion générale:
            Prévoir l'évolution des territoires se révèle être une tâche difficile. 
            Cette difficulté est d'autant plus grande quand on ne regarde que le passé, et plus encore un seul aspect du passé (économie).
            L'idéal pour nous aurait été d'avoir des variables directrices, corrélées au dynamisme passé, et disponibles dans le futur.
            Cela aurait pu être les investissements/subventions dans les infrastructures et les établissements, l'évolution démographique ... mais ces projections ne sont pas disponibles.
            
            Néanmoins nos modèles concernant les créations nettes des établissements et leur durée de vie ont donné de bons résultats, avec des ratios réel/prédiction assez proches de 1.
                                
            ### Conclusion concernant les modèles de prédiction:
            Nous pouvons dire que notre objectif de prédire le dynamisme des territoires n’est que partiellement atteint, car nous n’avons 
            modélisé que la dimension économique des territoires, et d’une façon étroite (volumes et durée de vie des établissements).
            La raison est liée à l'absence de séries temporelles sur les autres dimensions du dynamisme des territoires, comme la démographie, le secteur du BTP, les infrastructures publiques, l'emploi ...
            <br>
            Même si des séries existent pour certaines de ces thématiques, elles ne sont pas aussi longues, ou pas aussi fines (dans le temps et l'espace) que notre série principale qui porte sur les créations des établissements.
                    
            
            ### Conclusion concernant le clustering:
            Le clustering et particulièrement l'application des modèles de réduction (T-SNE et UMAP) nous montrent un regroupement homogène des communes selon leurs
            caractéristiques socio-économiques, démographiques et immobilières.
            <br>
            En général les communes les plus riches (revenus des ménages) et les plus chères (immobilier), sont celles qui connaissent la démographie la plus dynamique, les taux d'emploi et les taux de diplômés (>bac +2) les plus élevés.<br>
            Cependant nous ne voyons pas de corrélation de ces indicateurs avec les volumes de création et la durée de vie des établissements.
                    
            ### Perspectives:
            Les modèles que nous avons développé pourraient être utiles dans divers secteurs, particulièrement si nous ajoutons des couches de précision pour discriminer les prédictions par type d’activité exercée, et tranches d’effectifs employés.
            Cette extension peut être envisagée à condition de trouver au moins une série temporelle sur la même structure que la base Sirène, et fortement corrélée à cette dernière.

            Les secteurs d’activité concernés peuvent être:
            ##### <u>1. Agences immobilières:</u>
            * Prospection et acquisition : <br>
            Les agences immobilières peuvent utiliser ces modèles pour identifier les zones géographiques où de nouvelles entreprises sont susceptibles de se créer. Cela leur permettrait de cibler des acheteurs particuliers pour des projets résidentiels, ou des investisseurs potentiels pour l'achat ou la location de biens immobiliers commerciaux.

            * Évaluation des tendances du marché : <br> 
            Les modèles peuvent aider à anticiper la demande pour des espaces commerciaux dans certaines régions, permettant aux agences de mieux ajuster leur offre de service.


            ##### <u>2. Bâtiment et Travaux Publics (BTP):</u>
            * Planification de projets de construction: <br>
            Les entreprises de construction peuvent utiliser les modèles pour prévoir les besoins en nouvelles infrastructures (logements, bureaux, commerces, usines, entrepôts, etc.) dans les territoires où une croissance économique est attendue.

            * Optimisation des ressources : <br>
            Les prédictions peuvent aider à mieux allouer les ressources (main-d'œuvre, matériaux, équipements) en fonction des zones où une hausse de l'activité économique est prévue.

            * Anticipation des besoins en infrastructures publiques: <br>
            Les collectivités locales et les entreprises de BTP peuvent planifier à l'avance les infrastructures publiques (routes, ponts, réseaux de transport) en fonction des zones de développement économique.


            ##### <u>3. Implantation de nouvelles entreprises:</u>
            * Choix de l'emplacement:<br>
            Les entrepreneurs et les entreprises cherchant à s'implanter peuvent utiliser les prédictions, en complément des outils de géomarketing, pour identifier les zones les plus prometteuses, en termes de croissance économique, de demande de marché et de concurrence.

            * Évaluation des risques: <br>
            Les investisseurs et les entreprises peuvent évaluer les risques économiques régionaux, en identifiant les zones où la création d'entreprises est susceptible de ralentir ou d'accélérer, influençant ainsi leurs décisions d'investissement.

            * Stratégies de croissance: <br>
            Les entreprises existantes peuvent utiliser le modèle pour planifier le développement de leur activité, en identifiant les secteurs économiques les plus dynamiques.

            ##### <u>4. Secteur financier (banques, assurances):</u>
            * Analyse de crédit : <br>
            Les banques et institutions financières peuvent utiliser les prévisions pour évaluer la viabilité des prêts aux entreprises dans certains territoires ou secteurs d'activité, en tenant compte des tendances économiques locales.

            * Souscription d'assurance: <br>
            Les compagnies d'assurance peuvent ajuster leurs primes, selon les territoires où la création d'entreprises et leur durée de vie est en augmentation ou en déclin.

            ##### <u>5. Planification urbaine et développement économique:</u>
            * Développement local : <br>
            Les communes et régions peuvent utiliser le modèle pour planifier le développement économique local, en se concentrant sur les territoires et secteurs d’activité où une augmentation des créations d'entreprises est prévue.

            * Attraction d'investissements: <br>
            Les agences de développement économique peuvent utiliser ces informations pour attirer des investissements dans des secteurs d’activité ou des territoires spécifiques, qui montrent des prévisions favorables de création d'entreprises.

            * Développement des infrastructures: <br>
            Les autorités locales peuvent prévoir les besoins en infrastructures (transport, éducation, santé) en fonction de la croissance attendue du nombre d'entreprises.

            ##### <u>6. Conseil et analyse économique:</u>
            Les cabinets de conseil peuvent utiliser les prédictions pour fournir des conseils aux entreprises, en les aidant à identifier les opportunités de marché et à planifier leur expansion.

            ##### <u>7. Emploi et recrutement:</u>            
            Ces modèles peuvent aider les agences de recrutement à cibler efficacement les zones à fort potentiel, anticiper les besoins en main d’œuvre, 
            optimiser leurs propres ressources selon les zones et les profils les plus en demande, et développer des partenariats à long terme avec leurs clients.
            
            
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()