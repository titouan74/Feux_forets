from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import folium
import geopandas as gpd
import numpy as np
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

#cd Document

st.set_page_config(
    layout="wide",  # Peut être 'centered' ou 'wide', 'wide' est plus spacieux
)

@st.cache_data
def load_data():
    return pd.read_csv("Fires.csv")
df1 = load_data()

@st.cache_data
def load_data():
    return pd.read_csv("MergedFinal.csv")
df = load_data()


###################################################################SIDEBAR

st.title("# Prédiction des causes de feux de forets aux Etats Unis")
st.sidebar.header(" DATA SCIENTEST")
st.sidebar.write('\n')
st.sidebar.title("# Sommaire")
pages=["Introduction","Exploration", "Pre-processing", "Modélisation",'Optimisation',"Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write('\n')
st.sidebar.write('\n')
st.sidebar.write('\n')
st.sidebar.write('\n')
st.sidebar.text("LYMER Carole")
st.sidebar.text("OTTINGER Titouan")
st.sidebar.text("YEROU Sarra")
st.sidebar.image("DS.png")

###################################################################INTRODUCTION

if page == pages[0]:
    st.markdown('<h1 style="color: red;">INTRODUCTION</h1>', unsafe_allow_html=True)

    st.write('Ce projet consiste à analyser les feux de forêts aux Etats Unis et à prédire leur cause. En effet, comprendre les paramètres dapparition des feux de forets permettrait en théorie de pouvoir définir une stratégie de prévention')
    st.write('Le jeu de données rassemble ainsi les feux de forêts aux Etats Unis de 1992 à 2009.')
    st.write('Le fichier est téléchargeable au lien suivant: https://www.kaggle.com/rtatman/188-million-us-wildfires')

    st.image("mur-feu.png", use_column_width=True)

    st.write('### Dimension du dataset:')
    st.write(df1.shape)

    st.write('### Aperçu des premières lignes:')
    st.dataframe(df1.head(10))

    st.write('### Nombre de doublons:')
    st.write(df1.duplicated().sum())

    st.write('### Cellules vides et Types des variables:')
    info_dict = {
            "Non-null count": df1.count(),
            "Data type": df1.dtypes,
            "Nb modalités": df1.nunique()
    }
    info_df = pd.DataFrame(info_dict)
    st.dataframe(info_df)

    st.write('### Conclusion:')
    st.write('- Peu de datas réellement intéressantes (CAUSE, YEAR, DISCOVERY, CONTAINMENT, SIZE, OWNER, COORDONNEES, COUNTY/STATE) alors que le dataset comprend 40 variables')
    st.write('- Duo de variables correspondant aux mêmes informations text (DESCR) vs numérique (CODE)')
    st.write('- Des types incorrects (tout est Object, INT ou FLOAT)')
    st.write('- Beaucoup de cellules vides, notamment dans des variables identifiantes non pertinentes')
    st.write('\n')
    st.write('Le dataset a donc nécessairement besoin de deux étapes:')
    st.write('- NETTOYAGE (suppression de variables,modification de format)')
    st.write('- ENRICHISSEMENT(nouvelles data)')


###################################################################EXPLORATION

if page == pages[1]:
    st.markdown('<h1 style="color: red;">EXPLORATION</h1>', unsafe_allow_html=True)
   
    choix = ["Selectionner","Visualisation générale", "Visualisation carte des USA par cause", "Visualisation carte des USA par taille et cause" , "Visualisation météo", "Visualisation des campings", "Visualisation de la criminalité", "Visualisation des éléments géographiques"]
    option = st.selectbox('Choix de la visualisation', choix)
    st.write('La visualisation choisie est :', option)

###################################################################VISUALISATION GENERALE

    if option == "Visualisation générale":
        
                # Nombre d'incendies par cause aux Etats-Unis
        import plotly.express as px
        compte_causes = df['STAT_CAUSE_DESCR'].value_counts().reset_index()
        compte_causes.columns = ['Cause', 'Nombre d\'Incendies']
        # Créer un Histogramme
        fig = px.bar(compte_causes,
                    x='Cause',
                    y='Nombre d\'Incendies',
                    title='Nombre d\'Incendies par Cause aux États-Unis',
                    labels={'Cause': 'Cause d\'Incendie', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    color='Nombre d\'Incendies',
                    text='Nombre d\'Incendies')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)


        # Nombre d'incendies par cause aux Etats-Unis toutes années confondues
        # Compter le nombre d'incendies par mois
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        df['Month'] = df['DISCOVERY_COMPLETE'].dt.month
        nombre_incendies_par_mois = df['Month'].value_counts().sort_index()
        # Créer un DataFrame pour la visualisation
        df_mois = pd.DataFrame(nombre_incendies_par_mois).reset_index()
        df_mois.columns = ['Month', 'Nombre d\'Incendies']
        df_mois['Mois'] = pd.to_datetime(df_mois['Month'], format='%m').dt.strftime('%B')
        # Créer un histogramme
        fig = px.bar(df_mois,
                    x='Mois',
                    y='Nombre d\'Incendies',
                    title='Nombre d\'Incendies par Mois aux États-Unis toutes années confondues',
                    labels={'Mois': 'Mois', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    text='Nombre d\'Incendies',
                    category_orders={"Mois": ["January", "February", "March", "April", "May",
                                                "June", "July", "August", "September",
                                                "October", "November", "December"]})
        st.plotly_chart(fig)


        # Nombre d'incendies par Etat aux Etats-Unis
        # Compter le nombre d'incendies par état
        incendies_par_etat = df['STATE'].value_counts().reset_index()
        incendies_par_etat.columns = ['État', 'Nombre d\'Incendies']
        # Créer la carte de chaleur
        fig = px.choropleth(incendies_par_etat,
                            locations='État',
                            locationmode='USA-states',
                            color='Nombre d\'Incendies',
                            scope='usa',
                            color_continuous_scale='Viridis',
                            title='Nombre d\'Incendies par État aux États-Unis')
        st.plotly_chart(fig)

        #PROPORTION DES FEUX CAUSES PAR LA FOUDRE PAR TAILLE:
        # Total feux causés par la foudre par taille:
        total_fire_counts = df.groupby('FIRE_SIZE_CLASS').size().reset_index(name='Total_Fires')
        lightning_counts = df[df['STAT_CAUSE_DESCR'] == 'Lightning'].groupby('FIRE_SIZE_CLASS').size().reset_index(name='Lightning_Fires')
        combined_counts = pd.merge(total_fire_counts, lightning_counts, on='FIRE_SIZE_CLASS', how='left').fillna(0)
        # Proportion:
        combined_counts['Proportion_Lightning'] = (combined_counts['Lightning_Fires'] / combined_counts['Total_Fires']) * 100
        # Régression:
        x = np.arange(len(combined_counts))
        y = combined_counts['Proportion_Lightning']
        slope, intercept = np.polyfit(x, y, 1)
        predictions = slope * x + intercept
        fig.add_scatter(
            x=combined_counts['FIRE_SIZE_CLASS'],
            y=predictions,
            mode='lines',
            name='Droite de régression',
            line=dict(color='red', width=2))
        # Graph
        fig = px.bar(
            combined_counts,
            x='FIRE_SIZE_CLASS',
            y='Proportion_Lightning',
            title='Proportion des feux causés par la foudre par classe de taille',
            labels={'Proportion_Lightning': 'Proportion de feux causés par la foudre (%)', 'FIRE_SIZE_CLASS': 'Taille des feux'},)
        # Ajouter % sur les barres:
        for index, row in combined_counts.iterrows():
            fig.add_annotation(
                x=row['FIRE_SIZE_CLASS'],
                y=row['Proportion_Lightning'] + 5,
                text=f"{row['Proportion_Lightning']:.2f}%",
                showarrow=False,
                font=dict(size=16))
        # Ajuster les axes pour avoir 0% à 100%
        fig.update_yaxes(range=[0, 100])
        # Afficher la figure
        st.plotly_chart(fig)


###################################################################SELECTIONNER       

    if option == "Selectionner":
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('En plus des opérations de modification et de nettoyage, le DataSet a été enrichi de nombreuses nouvelles variables:')
        st.write('- Données de températures et de précipitations')
        st.write('- Données liées aux crimes')
        st.write('- Données liées à la faune et à la flore (Ecorégions)')
        st.write('- Données de distance entre le feu et des éléments divers (camping, chemin de fer, villes etc.)')

###################################################################VISUALISATION METEO       

    if option == "Visualisation météo":

        df['TEMP_MOY'] = df['TEMP_MOY'].round(0)
        # Compter le nombre d'incendies par température moyenne
        nombre_incendies_par_temp = df.groupby('TEMP_MOY').size().reset_index(name='Nombre d\'Incendies')
        # Créer une courbe
        fig = px.line(nombre_incendies_par_temp,
                    x='TEMP_MOY',
                    y='Nombre d\'Incendies',
                    title='Nombre d\'Incendies en fonction de la Température Moyenne',
                    labels={'TEMP_MOY': 'Température Moyenne (°C)', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    markers=True)  # Ajoute des marqueurs aux points de données
        fig.update_layout(
            xaxis_title='Température Moyenne (°C)',
            yaxis_title='Nombre d\'Incendies',
        )
        st.plotly_chart(fig)


        # Nombre d'Incendies et Température Moyenne par Mois aux États-Unis
        # Calculer la température moyenne par mois
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        df['Month'] = df['DISCOVERY_COMPLETE'].dt.month
        nombre_incendies_par_mois = df['Month'].value_counts().sort_index()
        temp_moy_par_mois = df.groupby('Month')['TEMP_MOY'].mean()
        # Créer un DataFrame pour la visualisation
        df_mois = pd.DataFrame({
            'Month': nombre_incendies_par_mois.index,
            'Nombre d\'Incendies': nombre_incendies_par_mois.values,
            'Température Moyenne': temp_moy_par_mois.values
        })
        df_mois['Mois'] = pd.to_datetime(df_mois['Month'], format='%m').dt.strftime('%B')
        # Créer un graphique
        fig = px.bar(df_mois,
                    x='Mois',
                    y='Nombre d\'Incendies',
                    title='Nombre d\'Incendies et Température Moyenne par Mois aux États-Unis',
                    labels={'Mois': 'Mois', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    text='Nombre d\'Incendies',
                    category_orders={"Mois": ["January", "February", "March", "April", "May",
                                                "June", "July", "August", "September",
                                                "October", "November", "December"]})
        # Ajouter la température moyenne sur le même graphique
        fig.add_scatter(x=df_mois['Mois'], y=df_mois['Température Moyenne'],
                        mode='lines+markers', name='Température Moyenne',
                        yaxis='y2', line=dict(color='orange'))
        fig.update_layout(
            yaxis_title='Nombre d\'Incendies',
            yaxis2=dict(title='Température Moyenne (°C)', overlaying='y', side='right'),
        )
        st.plotly_chart(fig)


        # Nombre d'Incendies et Précipitations Moyennes par Mois aux États-Unis
        # Calculer la précipitation moyenne par mois
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        df['Month'] = df['DISCOVERY_COMPLETE'].dt.month
        nombre_incendies_par_mois = df['Month'].value_counts().sort_index()
        precip_moy_par_mois = df.groupby('Month')['PRECTOTCORR'].mean()
        # Créer un DataFrame pour la visualisation
        df_mois = pd.DataFrame({
            'Month': nombre_incendies_par_mois.index,
            'Nombre d\'Incendies': nombre_incendies_par_mois.values,
            'Précipitations Moyennes': precip_moy_par_mois.values
        })
        df_mois['Mois'] = pd.to_datetime(df_mois['Month'], format='%m').dt.strftime('%B')
        # Créer un graphique
        fig = px.bar(df_mois,
                    x='Mois',
                    y='Nombre d\'Incendies',
                    title='Nombre d\'Incendies et Précipitations Moyennes par Mois aux États-Unis',
                    labels={'Mois': 'Mois', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    text='Nombre d\'Incendies',
                    category_orders={"Mois": ["January", "February", "March", "April", "May",
                                                "June", "July", "August", "September",
                                                "October", "November", "December"]})
        # Ajouter les précipitations sur le même graphique
        fig.add_scatter(x=df_mois['Mois'], y=df_mois['Précipitations Moyennes'],
                        mode='lines+markers', name='Précipitations Moyennes (mm)',
                        yaxis='y2', line=dict(color='blue'))
        fig.update_layout(
            yaxis_title='Nombre d\'Incendies',
            yaxis2=dict(title='Précipitations Moyennes (mm)', overlaying='y', side='right'),
        )
        st.plotly_chart(fig)


        # 5 Principales Causes d'Incendies en Fonction des Précipitations aux États-Unis
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        df['Month'] = df['DISCOVERY_COMPLETE'].dt.month
        # Calculer le nombre d'incendies par précipitations et par cause et ordonner les causes par le nombre total d'incendies
        precip_causes = df.groupby(['PRECTOTCORR', 'STAT_CAUSE_DESCR']).size().reset_index(name='Nombre d\'Incendies')
        cause_order = precip_causes.groupby('STAT_CAUSE_DESCR')['Nombre d\'Incendies'].sum().sort_values(ascending=False).index.tolist()
        # Sélectionner les 5 principales causes pour chaque niveau de précipitation
        top_causes_per_precip = (
            precip_causes
            .sort_values(by='Nombre d\'Incendies', ascending=False)
            .groupby('PRECTOTCORR')
            .head(5)
        )
        # Créer un histogramme
        fig = px.bar(top_causes_per_precip,
                    x='PRECTOTCORR',
                    y='Nombre d\'Incendies',
                    color='STAT_CAUSE_DESCR',  # Séparer par cause d'incendie
                    title='5 Principales Causes d\'Incendies en Fonction des Précipitations aux États-Unis',
                    labels={'PRECTOTCORR': 'Précipitations (mm)', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    barmode='stack',
                    category_orders={'STAT_CAUSE_DESCR': cause_order[:5]})  # Définir l'ordre des causes
        fig.update_xaxes(range=[-0.5, 5.5])  # Ajuster selon la plage désirée
        st.plotly_chart(fig)


        # 5 Principales Causes d'Incendies en Fonction de la Température Moyenne aux États-Unis
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        df['Month'] = df['DISCOVERY_COMPLETE'].dt.month
        # Calculer le nombre d'incendies par température moyenne et par cause et ordonner les causes par le nombre total d'incendies
        temp_causes = df.groupby(['TEMP_MOY', 'STAT_CAUSE_DESCR']).size().reset_index(name='Nombre d\'Incendies')
        cause_order = temp_causes.groupby('STAT_CAUSE_DESCR')['Nombre d\'Incendies'].sum().sort_values(ascending=False).index.tolist()
        # Sélectionner les 5 principales causes pour chaque niveau de température
        top_causes_per_temp = (
            temp_causes
            .sort_values(by='Nombre d\'Incendies', ascending=False)
            .groupby('TEMP_MOY')
            .head(5)
        )
        # Créer un histogramme
        fig = px.bar(top_causes_per_temp,
                    x='TEMP_MOY',
                    y='Nombre d\'Incendies',
                    color='STAT_CAUSE_DESCR',  # Séparer par cause d'incendie
                    title='5 Principales Causes d\'Incendies en Fonction de la Température Moyenne aux États-Unis',
                    labels={'TEMP_MOY': 'Température Moyenne (°C)', 'Nombre d\'Incendies': 'Nombre d\'Incendies'},
                    barmode='stack',
                    category_orders={'STAT_CAUSE_DESCR': cause_order[:5]})  # Définir l'ordre des causes
        fig.update_xaxes(range=[-10.5, 40.5])  # Ajuster selon la plage désirée
        st.plotly_chart(fig)








###################################################################VISUALISATION CAMPINGS

    if option == "Visualisation des campings":
        st.write('# Visualisation des campings')
        




























###################################################################VISUALISATION CRIMINALITE

    if option == "Visualisation de la criminalité":
        st.write('# Visualisation de la criminalité')

###################################################################VISUALISATION GEOGRAPHIQUE

    if option == "Visualisation des éléments géographiques":


                #BOXPLOT COORDONNEES:
        # Styles de Seaborn
        sns.set(style="whitegrid")
        # Créer une figure avec des sous-graphiques pour la latitude et la longitude
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        # Latitude
        sns.boxplot(data=df, x='STAT_CAUSE_DESCR', y='LATITUDE', ax=axes[0])
        axes[0].set_title('Distribution des Latitudes selon les Causes')
        axes[0].set_xlabel('Cause')
        axes[0].set_ylabel('Latitude')
        axes[0].tick_params(axis='x', rotation=45)
        # Longitude
        sns.boxplot(data=df, x='STAT_CAUSE_DESCR', y='LONGITUDE', ax=axes[1])
        axes[1].set_title('Distribution des Longitudes selon les Causes')
        axes[1].set_xlabel('Cause')
        axes[1].set_ylabel('Longitude')
        axes[1].tick_params(axis='x', rotation=45)
        # Afficher:
        plt.tight_layout()
        st.pyplot(fig)


            #MEDIANE PLUS PROCHE DES RAILS:
        #Médiane des distances par cause
        median_df = df.groupby('STAT_CAUSE_DESCR')['Distance_railroad'].median().reset_index()
        # Graph:
        fig = px.bar(
            median_df,
            x='STAT_CAUSE_DESCR',
            y='Distance_railroad',
            title='Médiane des distances par cause',
            labels={'distance': 'Médiane des distances', 'STAT_CAUSE_DESCR': 'Causes'},)
        # Afficher le graphique
        st.plotly_chart(fig)


        #HISTO DES FEUX CAUSES PAR LES RAILS ET <1000 des rails:
        df_rail = df[df['STAT_CAUSE_DESCR'] == 'Railroad'].copy()
        df_rail = df_rail[df_rail['Distance_railroad'] <= 1000].copy()
        bins = list(range(0, 1100, 100))
        df_rail['Distance Bins'] = pd.cut(df_rail['Distance_railroad'], bins=bins, right=False)
        df_rail['Distance Bins'] = df_rail['Distance Bins'].astype(str)
        counts = df_rail.groupby(['Distance Bins']).size().reset_index(name='Count')
        # Graph:
        fig = px.bar(
            counts,
            x='Distance Bins',
            y='Count',
            title='Quantité de Feux par Intervalle de Distance (Proche des Rails)',
            labels={'Count': 'Nombre de Feux', 'Distance Bins': 'Distance des rails'},
            text='Count')  # Ajouter le nombre sur les barres)
        # Afficher:
        fig.update_layout(
            width=1000,
            height=600)
        st.plotly_chart(fig)


        #MEDIANE PLUS PROCHE DES VILLES:
        # médiane de Nearest_Distance par cause
        median_distances_by_cause = df.groupby('STAT_CAUSE_DESCR')['Distance_place'].median().reset_index()
        median_distances_by_cause.columns = ['STAT_CAUSE_DESCR', 'Median_Nearest_Distance']
        # Graph:
        fig = px.bar(
            median_distances_by_cause,
            x='STAT_CAUSE_DESCR',
            y='Median_Nearest_Distance',
            title='Médiane de la Distance au Lieu le Plus Proche par Cause',
            labels={'Median_Nearest_Distance': 'Médiane de la Distance', 'STAT_CAUSE_DESCR': 'Cause du Feu'},)
        fig.update_layout(xaxis_tickangle=-45)  # Incliner les étiquettes de l'axe x pour une meilleure lisibilité
        st.plotly_chart(fig)


        #AJOUT DES TAILLES:
        # # médiane de Nearest_Distance par taille:
        median_distances_by_size = df.groupby('FIRE_SIZE_CLASS')['Distance_place'].median().reset_index()
        median_distances_by_size.columns = ['FIRE_SIZE_CLASS', 'Median_Nearest_Distance']
        # Graph:
        fig = px.bar(
            median_distances_by_size,
            x='FIRE_SIZE_CLASS',
            y='Median_Nearest_Distance',
            title='Médiane de la Distance au Lieu le Plus Proche par Taille de Feu',
            labels={'Median_Nearest_Distance': 'Médiane de la Distance', 'FIRE_SIZE_CLASS': 'Taille de Feu'},)
        # Régression:
        median_distances_by_size['FIRE_SIZE_NUM'] = pd.factorize(median_distances_by_size['FIRE_SIZE_CLASS'])[0]
        coefficients = np.polyfit(median_distances_by_size['FIRE_SIZE_NUM'], median_distances_by_size['Median_Nearest_Distance'], 1)
        polynomial = np.poly1d(coefficients)
        x_values = median_distances_by_size['FIRE_SIZE_NUM']
        y_values = polynomial(x_values)
        fig.add_trace(go.Scatter(
            x=median_distances_by_size['FIRE_SIZE_CLASS'],
            y=y_values,
            mode='lines',
            name='Droite de Régression',
            line=dict(color='red')))
        # Afficher:
        fig.update_layout(xaxis_tickangle=-45)  # Incliner les étiquettes de l'axe x pour une meilleure lisibilité
        st.plotly_chart(fig)


        #NUAGE DE POINT SELON % COUVERTURE FORESTIERE:
        fire_counts = df['county'].value_counts().reset_index()
        fire_counts.columns = ['county', 'fire_count']
        merged_df = fire_counts.merge(df[['county', 'Percent Forest Cover']].drop_duplicates(), on='county')
        # Nuage de points:
        fig = px.scatter(
            merged_df,
            x='Percent Forest Cover',
            y='fire_count',
            text='county',
            title='Nombre de Feux par County en Fonction du Pourcentage de Couverture Forestière',
            labels={'Percent Forest Cover': 'Pourcentage de Couverture Forestière', 'fire_count': 'Nombre de Feux'},)
        #Afficher:
        fig.update_traces(textposition='top center')
        fig.update_layout(
            width=1000,
            height=600,)
        st.plotly_chart(fig)


        #HISTOGRAMME ECOREGION PAR CAUSE:
        fig = px.histogram(
            df,
            x='ECOL1',
            color='STAT_CAUSE_DESCR',
            title='Histogramme par ecoregion',
            barmode='overlay',  # ou group ou stack
            labels={'ECOL1': '% Ecorégion niveau 1'},)
        #Afficher:
        fig.update_layout(width=1200, height=600)
        st.plotly_chart(fig)


###################################################################VISUALISATION CARTE DES USA PAR CAUSE

    if option == "Visualisation carte des USA par cause":
        
            # CARTE US PAR CAUSE :
        fig = px.scatter_geo(
            df,
            lat='LATITUDE',
            lon='LONGITUDE',
            color='STAT_CAUSE_DESCR',
            hover_name='FIRE_SIZE',
            animation_frame='FIRE_YEAR',
            title='Emplacements des feux par cause avec filtre par année',
            labels={'STAT_CAUSE_DESCR': 'Cause des feux'},
            color_discrete_sequence=px.colors.qualitative.Set1,)
        # Mise à jour de la géographie
        fig.update_geos(
            scope='usa',
            showland=True,
            landcolor='lightgray',
            countrycolor='white')
        # Mise à jour de la mise en page
        fig.update_layout(
            width=1000,
            height=600)
        # Affichage dans Streamlit avec st.plotly_chart
        st.plotly_chart(fig)


###################################################################VISUALISATION CARTE DES USA PAR TAILLE ET CAUSE

    if option == "Visualisation carte des USA par taille et cause":


                # CARTE US PAR CAUSE ET TAILLE:
        fig = px.scatter_geo(
            df,
            lat='LATITUDE',
            lon='LONGITUDE',
            color='STAT_CAUSE_DESCR',
            size='FIRE_SIZE',
            hover_name='FIRE_SIZE',
            animation_frame='FIRE_YEAR',
            title='Emplacements des feux par cause, taille et un filtre sur lannée',
            labels={'STAT_CAUSE_DESCR': 'Cause des feux', 'FIRE_SIZE': 'Taille du feu'},
            color_discrete_sequence=px.colors.qualitative.Set1)
        # Ajuster aux US:
        fig.update_geos(
            scope='usa',
            showland=True,
            landcolor='lightgray',
            countrycolor='white')
        # Modifier taille:
        fig.update_layout(
            width=1000,
            height=600)
        # Afficher:
        st.plotly_chart(fig)







###################################################################VISUALISATION GENERALE










if page == pages[2]:
    st.markdown('<h1 style="color: red;">PRE-PROCESSING</h1>', unsafe_allow_html=True)

    variable_types = {
            'STAT_CAUSE_DESCR': 'Variable cible',
            'STATE': 'Variables cardinales',
            'county': 'Variables cardinales',
            'ECOL1': 'Variables cardinales',
            'ECOL2': 'Variables cardinales',
            'ECOL3': 'Variables cardinales',
            'OWNER_CODE': 'Variables cardinales',
            'Nearest_Place_Type': 'Variables cardinales',
            'DISCOVERY_DAY_OF_WEEK': 'Variables cardinales',
            'FIRE_SIZE_CLASS': 'Variables ordinales',
            'Distance_railroad': 'Variables quantitatives',
            'Distance_place': 'Variables quantitatives',
            'Nearest_City_Distance': 'Variables quantitatives',
            'distance_to_camp_km': 'Variables quantitatives',
            'T2M_MIN': 'Variables quantitatives',
            'T2M_MAX': 'Variables quantitatives',
            'PRECTOTCORR': 'Variables quantitatives',
            'TEMP_MOY': 'Variables quantitatives',
            'Percent Forest Cover': 'Variables quantitatives',
            'TOTAL_CRIMES': 'Variables quantitatives',
            'POP_MOY': 'Variables quantitatives',
            'DENS_MOY': 'Variables quantitatives',
            'CRIME_RATE_YEAR': 'Variables quantitatives',
            'TOTAL_CRIMES_GLOBAL': 'Variables quantitatives',
            'CRIME_RATE': 'Variables quantitatives',
            'LATITUDE': 'Variables circulaires',
            'LONGITUDE': 'Variables circulaires',
            'DISCOVERY_COMPLETE': 'Variables circulaires'
    }

    modalities_count = {col: df[col].nunique() for col in df.columns}

    variable_info = pd.DataFrame({
            'Nom de la variable': df.columns,
            'Type de la variable': [variable_types.get(col, 'Inconnu') for col in df.columns],
            'Nombre de modalités': [modalities_count[col] for col in df.columns]
    })

    modalities_count = df['STAT_CAUSE_DESCR'].value_counts()
    modalities_percentage = ((modalities_count / len(df)) * 100).sort_values(ascending = False)

    choices = st.multiselect("Le processus de normalisation et d'encodage dépend des types de variables", ["Variable cible", "Variables cardinales", "Variables ordinales", "Variables quantitatives", "Variables circulaires"])
    st.write(f"Vous avez sélectionné : {', '.join(choices)}")

    if choices:
        filtered_table = variable_info[variable_info['Type de la variable'].isin(choices)]
        st.dataframe(filtered_table, use_container_width=True)
    
    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.write("**La variable cible : les différentes causes des feux**")

    st.write(modalities_percentage.index[0], modalities_percentage.iloc[0],"%", "*--> Feu d'origine naturel avec une source d'énergie provenant de la foudre*")

    st.write(modalities_percentage.index[1], modalities_percentage.iloc[1],"%","*--> Origine criminel*")
    
    st.write(modalities_percentage.index[2], modalities_percentage.iloc[2],"%","*--> Feu avec des causes diverses*")

    st.write(modalities_percentage.index[3], modalities_percentage.iloc[3],"%","*--> Feu de camp mal contrôlé*")

    st.write(modalities_percentage.index[4], modalities_percentage.iloc[4],"%","*--> Brulage de déchet non maitrisé*")

    st.write(modalities_percentage.index[5], modalities_percentage.iloc[5],"%","*--> Feu causé par des enfants*")

    st.write(modalities_percentage.index[6], modalities_percentage.iloc[6],"%","*--> Utilisation d'un équipement (incident électrique, fuite, produits inflammable, accident etc.*")

    st.write(modalities_percentage.index[7], modalities_percentage.iloc[7],"%","*--> Mégot de cigarette mal éteint*")

    st.write(modalities_percentage.index[8], modalities_percentage.iloc[8],"%","*--> Retombées de débris enflammés ou explosion*")

    st.write(modalities_percentage.index[9], modalities_percentage.iloc[9],"%","*--> Feu provenant de l'échappement ou de freins défectueux des trains*")

    st.write(modalities_percentage.index[10], modalities_percentage.iloc[10],"%","*--> Vegetations/animaux touchant les lignes électriques ou lignes éléctriques qui tombent*")

    st.write(modalities_percentage.index[11], modalities_percentage.iloc[11],"%","*--> Incident dans un batiments d'habitation, commercial ou industriel*")

    st.write(modalities_percentage.index[12], modalities_percentage.iloc[12],"%","*--> Feu où l'origine n'a pas été determiné*")



########################################################################################################### MODELISATION 13 CLASSES

if page == pages[3]:
    st.markdown('<h1 style="color: red;">MODELISATION</h1>', unsafe_allow_html=True)

    # Initialisation de l'état de session si nécessaire
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None

        # Fonction pour réinitialiser
    def reset_selection():
        st.session_state.selected_class = None

        # Créer un bouton reset
    if st.button('Reset'):
        reset_selection()

    display = st.radio('Que souhaitez-vous faire ?', ('Montrer seulement les résultats', 'Charger les modèles complets'))
    if display == 'Montrer seulement les résultats':
        if st.button('13 classes : **Toutes** les causes'):
            st.write("Vous avez sélectionné 13 classes.")
            st.image('13classes1.png')
            st.image('13classes2.png')
        if st.button('11 classes : Causes **Divers** et **Undefined** exclus'):
            st.write("Vous avez sélectionné 11 classes.")
            st.image('11classes1.png')
            st.image('11classes2.png')
        if st.button('9 classes : Regroupement basé sur **matrice**: Arson/Powerline, Campfire/Smoking'):
            st.write("Vous avez sélectionné 9 classes.")
            st.image('9classes1.png')
            st.image('9classes2.png')
        if st.button('5 classes : Regroupement des causes liées aux **espaces residentiels**'):
            st.write("Vous avez sélectionné 5 classes.")
            st.image('5classes1.png')
            st.image('5classes2.png')
        if st.button('4 classes : Regroupement **Infrastructure**, Feu **humain**, Activité humaine **indirecte**'):
            st.write("Vous avez sélectionné 4 classes.")
            st.image('4classes1.png')
            st.image('4classes2.png')
        if st.button('2 classes : **Human** vs **Nature**'):
            st.write("Vous avez sélectionné 2 classes.")
            st.image('2classes1.png')
            st.image('2classes2.png')

    elif display == 'Charger les modèles complets':



        if st.button('13 classes : **Toutes** les causes'):
            st.write("Vous avez sélectionné 13 classes.")


            @st.cache_data
            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)





            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_13.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Arson',
                'Campfire',
                'Children',
                'Debris Burning',
                'Equipment Use',
                'Fireworks',
                'Lightning',
                'Miscellaneous',
                'Missing/Undefined',
                'Powerline',
                'Railroad',
                'Smoking',
                'Structure'
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit




    ########################################################################################################### MODELISATION 11 CLASSES


        if st.button('11 classes : Causes **Divers** et **Undefined** exclus'):
            st.write("Vous avez sélectionné 11 classes.")


            @st.cache_data
            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)


            #Suppression des classes non désirées:
            df = df[df['STAT_CAUSE_DESCR'] != 'Miscellaneous']
            df = df[df['STAT_CAUSE_DESCR'] != 'Missing/Undefined']





            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_11.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Arson',
                'Campfire',
                'Children',
                'Debris Burning',
                'Equipment Use',
                'Fireworks',
                'Lightning',
                'Powerline',
                'Railroad',
                'Smoking',
                'Structure'
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit







    ########################################################################################################### MODELISATION 9 CLASSES


        elif st.button('9 classes : Regroupement basé sur **matrice**: Arson/Powerline, Campfire/Smoking'):
            st.write("Vous avez sélectionné 9 classes.")

            @st.cache_data
            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)



            #Suppression des classes non désirées:
            #Suppression des classes non désirées:
            df = df[df['STAT_CAUSE_DESCR'] != 'Miscellaneous']
            df = df[df['STAT_CAUSE_DESCR'] != 'Missing/Undefined']

            df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].replace(
                    {
                    'Arson': 'Arson/Powerline',
                    'Campfire': 'Campfire/Smoking',
                    'Children': 'Children',
                    'Debris Burning': 'Debris Burning',
                    'Equipment Use': 'Equipment Use',
                    'Fireworks': 'Fireworks',
                    'Lightning': 'Lightning',
                    'Powerline': 'Arson/Powerline',
                    'Railroad': 'Railroad',
                    'Smoking': 'Campfire/Smoking',
                    'Structure': 'Structure'
                    }
            )



            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_9.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Arson/Powerline',
                'Campfire/Smoking',
                'Children',
                'Debris Burning',
                'Equipment Use',
                'Fireworks',
                'Lightning',
                'Railroad',
                'Structure'
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit
            



    ########################################################################################################### MODELISATION 5 CLASSES   


        elif st.button('5 classes : Regroupement des causes liées aux **espaces residentiels**'):
            st.write("Vous avez sélectionné 5 classes.")

            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)



            #Suppression des classes non désirées:
            df = df[df['STAT_CAUSE_DESCR'] != 'Miscellaneous']
            df = df[df['STAT_CAUSE_DESCR'] != 'Missing/Undefined']

            df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].replace(
                {
                'Arson': 'Arson/Powerline',
                'Campfire': 'Campfire/Smoking',
                'Children': 'Residential',
                'Debris Burning': 'Residential',
                'Equipment Use': 'Residential',
                'Fireworks': 'Residential',
                'Lightning': 'Lightning',
                'Powerline': 'Arson/Powerline',
                'Railroad': 'Railroad',
                'Smoking': 'Campfire/Smoking',
                'Structure': 'Residential'
                }
            )



            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_5.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Arson/Powerline',
                'Campfire/Smoking',
                'Lightning',
                'Railroad',
                'Residential',
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit

    ############################################################################################# MODELISATION 4 CLASSES   

        elif st.button('4 classes : Regroupement **Infrastructure**, Feu **humain**, Activité humaine **indirecte**'):
            st.write("Vous avez sélectionné 4 classes.")

            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)



            #Suppression des classes non désirées:
            df = df[df['STAT_CAUSE_DESCR'] != 'Miscellaneous']
            df = df[df['STAT_CAUSE_DESCR'] != 'Missing/Undefined']

            df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].replace(
                {
                'Arson': 'Human Fire',
                'Campfire': 'Human Fire',
                'Children': 'Human Activities',
                'Debris Burning': 'Human Fire',
                'Equipment Use': 'Infrastructure',
                'Fireworks': 'Human Activities',
                'Lightning': 'Lightning',
                'Powerline': 'Infrastructure',
                'Railroad': 'Infrastructure',
                'Smoking': 'Human Activities',
                'Structure': 'Infrastructure'
                }
            )



            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_4.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Human Activities',
                'Human Fire',
                'Infrastructure',
                'Lightning'
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit


    ############################################################################################# MODELISATION 2 CLASSES   
    # 
    #     
        elif st.button('2 classes : **Human** vs **Nature**'):
            st.write("Vous avez sélectionné 2 classes.")

            def load_data():
                return pd.read_csv("MergedFinal.csv")
            df = load_data()

            df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])

            #Création nouvelles colonnes pertinentes au lieu de la date complète:
            df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
            df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month

            #Suppression des colonnes et des doublons générés
            df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
            df = df.dropna(subset=['Discovery_year','Discovery_month'])
            df = df.drop_duplicates()

            # Suppression des valeurs rares dans 'county':
            frequencies = df['county'].value_counts()
            categories_rares = frequencies[frequencies < 40].index
            # Supprimer les lignes où 'county' est une catégorie rare, tout en gardant celles où 'county' est NaN
            df = df[~df['county'].isin(categories_rares) | df['county'].isna()]

            df = df.sort_values(by=['Discovery_year'],ascending = True)



            #Suppression des classes non désirées:
            df = df[df['STAT_CAUSE_DESCR'] != 'Miscellaneous']
            df = df[df['STAT_CAUSE_DESCR'] != 'Missing/Undefined']

            df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].replace(
                    
                {'Arson': 'Human',
                'Campfire': 'Human',
                'Children': 'Human',
                'Debris Burning': 'Human',
                'Equipment Use': 'Human',
                'Fireworks': 'Human',
                'Lightning': 'Nature',
                'Powerline': 'Human',
                'Railroad': 'Human',
                'Smoking': 'Human',
                'Structure': 'Human'
                }
                        
            )



            X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
            y = df['STAT_CAUSE_DESCR']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

            #Remplacer les NaN de crime rate par la médiane

            columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
            mean_values_train = X_train[
                X_train['STATE'].isin(['MD', 'VA'])
            ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()

            # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']

            # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
            X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
            X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
            X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']


            # Calculer la médiane de chaque colonne dans X_train
            median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()

            # Remplacer les NaN dans X_train par la médiane de chaque colonne
            X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            # Remplacer les NaN dans X_test par la médiane de chaque colonne
            X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)

            #Remplacement des N/A des nouvelles colonnes TITOUAN:

            #IL FAUT REMPLACER LES N/A DE COUNTY (notamment pour l'Alaska et New York qui ne fonctionne pas)
            # Trouver le county le plus fréquent pour chaque STATE
            most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            #Remplacer entraînement:
            X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
            #Remplacer test:
            X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)

            #Pour les quelques qui restent non expliqués:
            X_train = X_train.dropna(subset=['county'])
            X_test = X_test.dropna(subset=['county'])
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
            def replace_na_by_median_or_mode(df_train, df_test):
                #Calcul médiane:
                cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
                for col in cols_median:
                    #Calcul médiane sur entraînement
                    median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

                #Calcul mode:
                cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
                for col in cols_mode:
                    #Calcul mode sur entraînement
                    mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                    #Remplacer entrainement
                    df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                    #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                    df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)

                return df_train, df_test

            # Appliquer la fonction
            X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)

            #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
            #Mediane:
            columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']

            # Calcul des médianes uniquement sur l'ensemble d'entraînement
            median_values = X_train[columns_to_fill].median()

            # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

            #Mode:
            columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']

            mode_values = X_train[columns_to_fill].mode().iloc[0]

            # Appliquer le mode aux valeurs manquantes dans X_train et X_test
            X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
            X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)

            #Suppression de la colonne stat_cause_code:
            X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
            X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])





            #Encoder la variable cible et les variables trop diverses qui alourdiront le modèle avec un OHE:

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            # Afficher les correspondances des classes avec leur valeur numérique
            for idx, class_name in enumerate(le.classes_):
                print(f'La cause {class_name} = {idx}')

            cat = ['ECOL2','ECOL3']
            X_train = X_train.drop(cat,axis=1)
            X_test = X_test.drop(cat,axis=1)

            cat = ['STATE', 'county', 'ECOL1', 'OWNER_CODE']
            for column in cat:
                X_train[column] = le.fit_transform(X_train[column])
                X_test[column] = le.transform(X_test[column])

            #Encoder les variables cardinales:

            from sklearn.preprocessing import OneHotEncoder
            cat = ['Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
            #ohe = OneHotEncoder(drop="first", sparse_output=False)
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            # Appliquer OneHotEncoder sur X_train et X_test
            X_train_encoded = ohe.fit_transform(X_train[cat])
            X_test_encoded = ohe.transform(X_test[cat])
            # Convertir les résultats en DataFrame avec les bons noms de colonnes
            encoded_columns = ohe.get_feature_names_out(cat)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
            # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
            X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
            X_test = X_test.drop(columns=cat).join(X_test_encoded_df)

            #Encoder les variables ordinales:

            from sklearn.preprocessing import OrdinalEncoder
            import numpy as np
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
            X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
            X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)

            #Encoder les variables quantitatives:

            from sklearn.preprocessing import MinMaxScaler
            cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
            scaler = MinMaxScaler()
            X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:,cols] = scaler.transform(X_test[cols])

            #Encoder les coordonnées en gardant le lien géographique:

            X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
            X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
            X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
            X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))

            X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
            X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
            X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
            X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))

            X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
            X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)

            #Encoder les dates en gardant le cycle pour le mois:

            X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
            X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)

            X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
            X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)

            X_train.drop(columns=['Discovery_month'], inplace=True)
            X_test.drop(columns=['Discovery_month'], inplace=True)

            # 1. SMOTE (Synthetic Minority Over-sampling Technique)
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)



            model = joblib.load('model_2.pkl')
            y_pred = model.predict(X_test)

            # Faire des prédictions sur le jeu de test
            y_pred = model.predict(X_test)
            predictions = [round(value) for value in y_pred]

            # Prédictions sur le jeu d'entraînement
            y_train_pred = model.predict(X_train)
            predictions_train = [round(value) for value in y_train_pred]

            # Calcul de l'accuracy pour le jeu de test
            accuracy_test = accuracy_score(y_test, predictions)
            print("Accuracy sur le jeu de test: %.2f%%" % (accuracy_test * 100.0))

            # Calcul de l'accuracy pour le jeu d'entraînement
            accuracy_train = accuracy_score(y_train, predictions_train)
            print("Accuracy sur le jeu d'entraînement: %.2f%%" % (accuracy_train * 100.0))

            # Initialiser le LabelEncoder et l'ajuster sur y_train
            le = LabelEncoder()
            le.fit(y_train)  # Ajuster le LabelEncoder sur y_train

            # Prédictions en labels inversés
            y_pred_labels = le.inverse_transform(y_pred)

            # Créer un DataFrame pour obtenir l'accuracy par catégorie
            results = pd.DataFrame({
                'Actual': le.inverse_transform(y_test),
                'Predicted': y_pred_labels
            })

            # Calcul de l'accuracy par catégorie
            accuracy_per_class = results.groupby('Actual').apply(lambda x: (x['Predicted'] == x['Actual']).mean())

            # Calcul du nombre de feux pour chaque cause
            fire_count_per_class = results['Actual'].value_counts()

                # Définir les noms des classes
            class_names = [
                'Human',
                'Nature'
            ]

            # Fusionner l'accuracy et le nombre de feux dans un même DataFrame
            class_performance = pd.DataFrame({
                'Accuracy': accuracy_per_class,
                'Fire Count': fire_count_per_class
            })

            class_performance.index = class_names
            class_performance = class_performance.sort_values(by='Accuracy', ascending=False)

            # Affichage des résultats
            st.dataframe(class_performance)

            # Affichage des résultats
            st.write('**TEST ACCURACY:**',accuracy_score(y_test, predictions))      
            st.write('**TRAIN ACCURACY:**',accuracy_score(y_train, predictions_train))

            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, predictions)

            # Création de la figure pour la matrice de confusion
            fig, ax = plt.subplots(figsize=(10, 7))  # Créer un objet `fig` et `ax`
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel('Classe Prédites')
            ax.set_ylabel('Classe Réelles')
            ax.set_title('Matrice de Confusion')

            # Rotation des axes
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            # Afficher la figure sur Streamlit
            st.pyplot(fig)  # Utiliser `st.pyplot(fig)` pour afficher dans Streamlit
            



















    






























    















##############################################################################################PAGE 4 OPTIMISATION


if page == pages[4]:
        st.markdown('<h1 style="color: red;">OPTIMISATION</h1>', unsafe_allow_html=True)
        df['DISCOVERY_COMPLETE'] = pd.to_datetime(df['DISCOVERY_COMPLETE'])
        #Création nouvelles colonnes pertinentes au lieu de la date complète:
        df['Discovery_year'] = df['DISCOVERY_COMPLETE'].dt.year
        df['Discovery_month'] = df['DISCOVERY_COMPLETE'].dt.month
        #Suppression des colonnes et des doublons généré
        df = df.drop(columns=['OBJECTID','FIRE_YEAR','FIRE_SIZE','OWNER_DESCR','DURATION','CONT_COMPLETE','DISCOVERY_COMPLETE'])
        df = df.dropna(subset=['Discovery_year','Discovery_month'])
        df = df.drop_duplicates()
        #On remplace les modalités trop rares de county par une seule et même valeur "60000" pour éviter une classe manquante entre train et test
        frequencies = df['county'].value_counts()
        categories_rares = frequencies[frequencies < 15].index
        df['county'] = df['county'].apply(lambda x: '60000.0' if x in categories_rares else x)
        df['county'] = df['county'].astype('float')
        #Triage chronologique:
        df = df.sort_values(by=['Discovery_year'],ascending = True)

        # Mapping the causes to the new values
        cause_mapping = {
        'Arson': 'Humain Fire',
        'Campfire': 'Humain Fire',
        'Children': 'Humain Activities',
        'Debris Burning': 'Humain Fire',
        'Equipment Use': 'Infrastructure',
        'Fireworks': 'Humain Activities',
        'Lightning': 'Lightning',
        'Miscellaneous': None,
        'Missing/Undefined': None, 
        'Powerline': 'Infrastructure',
        'Railroad': 'Infrastructure',
        'Smoking': 'Humain Activities',
        'Structure': 'Infrastructure'
        }
        # Apply the mapping to the column 'STAT_CAUSE_DESCR'
        df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].map(cause_mapping)
        # Drop rows where the 'STAT_CAUSE_DESCR' column has a None value (i.e., Missing/Undefined)
        df = df.dropna(subset=['STAT_CAUSE_DESCR'])

        from sklearn.preprocessing import LabelEncoder
        # Encoder la variable cible
        le = LabelEncoder()
        df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])
        # Récupérer le mapping des classes après encodage
        class_mapping = dict(enumerate(le.classes_))
        # Liste des catégories d'intérêt
        categories_of_interest = {'Humain Fire', 'Humain Activities', 'Infrastructure', 'Lightning'}
        # Filtrer les indices associés aux catégories demandées
        category_indices = {label: idx for idx, label in class_mapping.items() if label in categories_of_interest}
        # Afficher les indices des catégories demandées
        print(category_indices)

        X = df.drop(['STAT_CAUSE_DESCR'], axis = 1)
        y = df['STAT_CAUSE_DESCR']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44, shuffle=False)

        columns_to_fill = ['TOTAL_CRIMES', 'POP_MOY', 'DENS_MOY', 'CRIME_RATE_YEAR', 'TOTAL_CRIMES_GLOBAL', 'CRIME_RATE']
        # Calcul des médianes uniquement sur l'ensemble d'entraînement
        median_values = X_train[columns_to_fill].median()
        # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
        X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
        X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)

        # Calculer la moyenne pour T2M_MIN, T2M_MAX, TEMP_MOY et PRECTOTCORR pour les états MD et VA dans X_train
        mean_values_train = X_train[
        X_train['STATE'].isin(['MD', 'VA'])
        ][['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].mean()
        # Appliquer les moyennes calculées à STATE = 'DC' dans X_train
        X_train.loc[X_train['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
        X_train.loc[X_train['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
        X_train.loc[X_train['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
        X_train.loc[X_train['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']
        # Appliquer les mêmes moyennes à STATE = 'DC' dans X_test
        X_test.loc[X_test['STATE'] == 'DC', 'T2M_MIN'] = mean_values_train['T2M_MIN']
        X_test.loc[X_test['STATE'] == 'DC', 'T2M_MAX'] = mean_values_train['T2M_MAX']
        X_test.loc[X_test['STATE'] == 'DC', 'TEMP_MOY'] = mean_values_train['TEMP_MOY']
        X_test.loc[X_test['STATE'] == 'DC', 'PRECTOTCORR'] = mean_values_train['PRECTOTCORR']
        # Calculer la médiane de chaque colonne dans X_train
        median_values = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].median()
        # Remplacer les NaN dans X_train par la médiane de chaque colonne
        X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_train[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)
        # Remplacer les NaN dans X_test par la médiane de chaque colonne
        X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']] = X_test[['T2M_MIN', 'T2M_MAX', 'TEMP_MOY', 'PRECTOTCORR']].fillna(median_values)
        #Remplacement des N/A des nouvelles colonnes TITOUAN:
        # Trouver le county le plus fréquent pour chaque STATE
        most_frequent_county_train = X_train.groupby('STATE')['county'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        #Remplacer entraînement:
        X_train['county'] = X_train.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
        #Remplacer test:
        X_test['county'] = X_test.apply(lambda row: most_frequent_county_train[row['STATE']] if pd.isna(row['county']) else row['county'], axis=1)
        #Remplacement des NaN par la médiane et le mode selon les colonnes, par county/cause:
        def replace_na_by_median_or_mode(df_train, df_test):
            #Calcul médiane:
            cols_median = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
            for col in cols_median:
                #Calcul médiane sur entraînement
                median_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].median()
                #Remplacer entrainement
                df_train[col] = df_train.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                df_test[col] = df_test.apply(lambda row: median_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in median_values.index else row[col], axis=1)

            #Calcul mode:
            cols_mode = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type']
            for col in cols_mode:
                #Calcul mode sur entraînement
                mode_values = df_train.groupby(['county', 'STAT_CAUSE_CODE'])[col].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                #Remplacer entrainement
                df_train[col] = df_train.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) else row[col], axis=1)
                #Remplacer test si NaN ET couple county/cause dans le jeu d'entrainment existe
                df_test[col] = df_test.apply(lambda row: mode_values.loc[(row['county'], row['STAT_CAUSE_CODE'])] if pd.isna(row[col]) and (row['county'], row['STAT_CAUSE_CODE']) in mode_values.index else row[col], axis=1)
            return df_train, df_test
        # Appliquer la fonction
        X_train, X_test = replace_na_by_median_or_mode(X_train, X_test)
        #Reste tout de même quelques NaN non expliquées. Pour cela nous allons remplacer par la mediane/mode basique:
        #Mediane:
        columns_to_fill = ['Percent Forest Cover', 'Distance_railroad', 'Distance_place', 'Nearest_City_Distance']
        # Calcul des médianes uniquement sur l'ensemble d'entraînement
        median_values = X_train[columns_to_fill].median()
        # Appliquer les médianes aux valeurs manquantes dans X_train et X_test
        X_train[columns_to_fill] = X_train[columns_to_fill].fillna(median_values)
        X_test[columns_to_fill] = X_test[columns_to_fill].fillna(median_values)
        #Mode:
        columns_to_fill = ['ECOL1', 'ECOL2', 'ECOL3', 'Nearest_Place_Type'] 
        mode_values = X_train[columns_to_fill].mode().iloc[0]
        # Appliquer le mode aux valeurs manquantes dans X_train et X_test
        X_train[columns_to_fill] = X_train[columns_to_fill].fillna(mode_values)
        X_test[columns_to_fill] = X_test[columns_to_fill].fillna(mode_values)
        #Suppression de la colonne stat_cause_code:
        X_test = X_test.drop(columns=['STAT_CAUSE_CODE'])
        X_train = X_train.drop(columns=['STAT_CAUSE_CODE'])

        #Encoder la variable cible:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Liste des colonnes pour LabelEncoding
        cat_le = ['STATE', 'county', 'ECOL1', 'ECOL2', 'ECOL3']
        # Dictionnaire pour sauvegarder les encodeurs de chaque colonne
        label_encoders = {}
        # Appliquer LabelEncoder sur X_train et gérer les catégories inconnues dans X_test
        for col in cat_le:
            le = LabelEncoder()
            # Ajuster l'encodeur sur les données d'entraînement
            X_train[col] = le.fit_transform(X_train[col])
            # Vérifier les catégories inconnues dans X_test
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else "unknown")
            # Ajouter "unknown" comme nouvelle catégorie et transformer X_test
            le.classes_ = np.append(le.classes_, "unknown") # Ajouter une catégorie "unknown"
            X_test[col] = le.transform(X_test[col])
            # Sauvegarder l'encodeur pour la colonne
            label_encoders[col] = le

        #Encoder les variables cardinales:
        from sklearn.preprocessing import OneHotEncoder
        cat = ['OWNER_CODE','Nearest_Place_Type','DISCOVERY_DAY_OF_WEEK']
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        # Appliquer OneHotEncoder sur X_train et X_test
        X_train_encoded = ohe.fit_transform(X_train[cat])
        X_test_encoded = ohe.transform(X_test[cat])
        # Convertir les résultats en DataFrame avec les bons noms de colonnes
        encoded_columns = ohe.get_feature_names_out(cat)
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
        # Supprimer les colonnes d'origine et ajouter les nouvelles colonnes encodées
        X_train = X_train.drop(columns=cat).join(X_train_encoded_df)
        X_test = X_test.drop(columns=cat).join(X_test_encoded_df)
        #Encoder les variables ordinales:
        from sklearn.preprocessing import OrdinalEncoder
        import numpy as np
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[["A", "B", "C", "D", "E", "F", "G"]])
        X_train['FIRE_SIZE_CLASS'] = oe.fit_transform(X_train[['FIRE_SIZE_CLASS']]).astype(float)
        X_test['FIRE_SIZE_CLASS'] = oe.transform(X_test[['FIRE_SIZE_CLASS']]).astype(float)
        #Encoder les variables quantitatives:
        from sklearn.preprocessing import MinMaxScaler
        cols = ['Distance_railroad','Distance_place','Nearest_City_Distance','distance_to_camp_km','T2M_MIN','T2M_MAX','PRECTOTCORR','TEMP_MOY','Percent Forest Cover','TOTAL_CRIMES','POP_MOY','DENS_MOY','CRIME_RATE_YEAR','TOTAL_CRIMES_GLOBAL','CRIME_RATE']
        scaler = MinMaxScaler()
        X_train.loc[:,cols] = scaler.fit_transform(X_train[cols])
        X_test.loc[:,cols] = scaler.transform(X_test[cols])
        #Encoder les coordonnées en gardant le lien géographique:
        X_train['latitude_sin'] = np.sin(np.radians(X_train['LATITUDE']))
        X_train['latitude_cos'] = np.cos(np.radians(X_train['LATITUDE']))
        X_train['longitude_sin'] = np.sin(np.radians(X_train['LONGITUDE']))
        X_train['longitude_cos'] = np.cos(np.radians(X_train['LONGITUDE']))
        X_test['latitude_sin'] = np.sin(np.radians(X_test['LATITUDE']))
        X_test['latitude_cos'] = np.cos(np.radians(X_test['LATITUDE']))
        X_test['longitude_sin'] = np.sin(np.radians(X_test['LONGITUDE']))
        X_test['longitude_cos'] = np.cos(np.radians(X_test['LONGITUDE']))
        X_train = X_train.drop(['LATITUDE', 'LONGITUDE'], axis=1)
        X_test = X_test.drop(['LATITUDE', 'LONGITUDE'], axis=1)
        #Encoder les dates en gardant le cycle pour le mois:
        X_train['Discovery_month_sin'] = np.sin(2 * np.pi * X_train['Discovery_month'] / 12)
        X_train['Discovery_month_cos'] = np.cos(2 * np.pi * X_train['Discovery_month'] / 12)
        X_test['Discovery_month_sin'] = np.sin(2 * np.pi * X_test['Discovery_month'] / 12)
        X_test['Discovery_month_cos'] = np.cos(2 * np.pi * X_test['Discovery_month'] / 12)
        X_train.drop(columns=['Discovery_month'], inplace=True)
        X_test.drop(columns=['Discovery_month'], inplace=True)
        from collections import Counter
        from imblearn.over_sampling import RandomOverSampler
        # Affichage de la distribution initiale des classes
        print("Distribution des classes avant Over Sampling :", Counter(y_train))
        # Application de Random Over Sampler
        ros = RandomOverSampler(random_state=44)
        X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        # Affichage de la distribution après Over Sampling
        print("Distribution des classes après Over Sampling :", Counter(y_train_ros))

        from sklearn.utils import resample
        # Sous-échantillonnage pour limiter la taille totale des données
        subset_size = 200000 # Nombre total d'échantillons souhaité 
        X_train_subset, y_train_subset = resample(X_train_ros, y_train_ros, 
        n_samples=subset_size, 
        random_state=42)
        # Vérification de la distribution après sous-échantillonnage
        print("Distribution des classes après sous-échantillonnage :", Counter(y_train_subset))

        import xgboost as xgb
        import numpy as np
        from sklearn.metrics import accuracy_score
        # Conversion des données en DMatrix pour XGBoost
        dtrain = xgb.DMatrix(X_train_subset, label=y_train_subset)
        dtest = xgb.DMatrix(X_test, label=y_test)
        # Définir les paramètres pour XGBoost
        params = {
        'objective': 'multi:softmax', # Classification multiclasse
        'num_class': len(np.unique(y_train_subset)), # Nombre de classes
        'max_depth': 10, # Profondeur des arbres
        'eta': 0.01, # Taux d'apprentissage
        'eval_metric': 'merror', # Erreur de classification
        'subsample': 0.9,
        'min_child_weight': 1
        }
        # Entraînement du modèle XGBoost
        num_round = 2000 # Nombre d'arbres
        bst = xgb.train(params, dtrain, num_round)
        # Prédiction sur les données de test
        y_pred = bst.predict(dtest)
        # Calcul de la précision (accuracy)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision du modèle XGBoost : {accuracy:.4f}")

        from sklearn.metrics import accuracy_score, classification_report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        import shap
        explainer = shap.Explainer(bst, X_train[:100])
        shap_values = explainer(X_test[:100])
        shap.summary_plot(shap_values[:,:,0], X_test[:100], max_display=55) 
        shap.summary_plot(shap_values[:,:,1], X_test[:100], max_display=55) 
        shap.summary_plot(shap_values[:,:,2], X_test[:100], max_display=55) 
        shap.summary_plot(shap_values[:,:,3], X_test[:100], max_display=55) 

        #display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Classification Report'))
        #if display == 'Accuracy':
        #st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # Affichage de l'accuracy
        #elif display == 'Classification Report':
        #report_dict = classification_report(y_test, y_pred, output_dict=True)
        #report_df = pd.DataFrame(report_dict).transpose()
        #st.dataframe(report_df) 
        # Interface Streamlit
        display = st.radio("Que souhaitez-vous montrer ?", 
        ('Accuracy', 'Classification Report', 'SHAP Summary Plots'))
        if display == 'Accuracy':
            st.write(f"Accuracy: {accuracy:.4f}") 
        elif display == 'Classification Report':
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df)
        elif display == 'SHAP Summary Plots':
        # Afficher les 4 Summary Plots dans Streamlit
            for i in range(4): # Supposant que tu as 4 classes
                st.write(f"SHAP Summary Plot pour la classe {i}")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values[:, :, i], X_test[:100], max_display=55, show=False)
                st.pyplot(fig) # Afficher le graphique dans Streamlit



        
    
        


if page == pages[5]:
    st.markdown('<h1 style="color: red;">CONCLUSION</h1>', unsafe_allow_html=True)
