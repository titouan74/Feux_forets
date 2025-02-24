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

#streamlit run projet.py

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
   
    choix = ["Selectionner","Visualisation générale", "Visualisation carte des USA", "Visualisation météo", "Visualisation des campings", "Visualisation de la criminalité", "Visualisation des éléments géographiques"]
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

        #Démarrage des visualisation pour établir un lien entre les feux et les campings
        df_total_fires_state = df.groupby(['STATE']).size().reset_index(name='FIRE_COUNT')
        df_merged=df_camp.merge(df_total_fires_state, on='STATE', how='left')

        fig = px.choropleth(
            data_frame=df_merged,
            locations='STATE',
            locationmode='USA-states',
            color='FIRE_COUNT',
            color_continuous_scale='YlOrRd',
            scope='usa',
            labels={'FIRE_COUNT': 'nombre de feux'},  # Renomme la légende de la couleur
            title='Nombre de feux de forêt et localisation des camping',
        )

        # Ajouter des points pour le nombre de feux avec les coordonnées latitude et longitude
        fig.add_trace(go.Scattergeo(
            locationmode='USA-states',
            lat=df['lat'],  # Colonne de latitude
            lon=df['lon'],  # Colonne de longitude
            text=df['camp'],  # Texte à afficher lors du survol
            mode='markers',
            marker=dict(
                #size=df['COUNT'] / 1850,  # Réduire la taille des points (divisé par 20)
                color='green',  # Couleur des points
                opacity=0.4,
                line=dict(width=0.3, color='black'),
            ),
            name='Nombre de camping'  # Nom de la légende
        ))

        # Mettre à jour la mise en page pour inclure un fond et d'autres options
        fig.update_layout(
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='lightgrey',
                subunitcolor='blue',
            ),
        )

        # Afficher la figure
        st.plotly_chart(fig)

        #Nb de feux de fôrets et visualisation des campings
        df_camp_fire_state=df.groupby('STATE')['FIRE_COUNT'].size().reset_index(name='COUNT')
        df_camp_fire_state = df.groupby('STATE').agg(
            COUNT=('STATE', 'size'),         # Compte le nombre de lignes
            FIRE_COUNT=('FIRE_COUNT', 'first')  # Somme de la colonne FIRE_COUNT
        ).reset_index()
        df_camp_fire_state

        # Créer un dictionnaire avec les coordonnées (latitude, longitude) de chaque État
        state_coords = {
            'AL': (32.806671, -86.791397),
            'AK': (61.370716, -152.404419),
            'AZ': (33.729759, -111.431221),
            'AR': (34.969704, -92.373123),
            'CA': (36.116203, -119.681564),
            'CO': (39.059811, -105.311104),
            'CT': (41.597782, -72.755371),
            'DE': (39.318523, -75.507141),
            'FL': (27.766279, -81.686783),
            'GA': (33.040619, -83.643074),
            'HI': (21.094318, -157.498337),
            'ID': (44.240459, -114.478828),
            'IL': (40.349457, -88.986137),
            'IN': (39.849426, -86.258278),
            'IA': (42.011539, -93.210526),
            'KS': (39.063946, -98.387207),
            'KY': (37.668140, -84.670067),
            'LA': (31.169546, -91.867805),
            'ME': (45.367584, -69.381927),
            'MD': (39.063946, -76.802101),
            'MA': (42.230171, -71.530106),
            'MI': (43.326618, -84.536095),
            'MN': (45.694454, -93.900192),
            'MS': (32.741646, -89.678696),
            'MO': (38.456085, -92.288368),
            'MT': (46.921925, -110.454353),
            'NE': (41.492537, -99.901810),
            'NV': (38.502503, -117.023060),
            'NH': (43.193852, -71.572395),
            'NJ': (40.298904, -74.521011),
            'NM': (34.840515, -106.248482),
            'NY': (42.165726, -74.948051),
            'NC': (35.630066, -79.806419),
            'ND': (47.528912, -99.784012),
            'OH': (40.388783, -82.764915),
            'OK': (35.565342, -96.928917),
            'OR': (43.933, -120.558),
            'PA': (40.590752, -77.209755),
            'RI': (41.680893, -71.511780),
            'SC': (33.856892, -80.945007),
            'SD': (44.299782, -99.438828),
            'TN': (35.747845, -86.692345),
            'TX': (31.054487, -97.563461),
            'UT': (40.150032, -111.862434),
            'VT': (44.045876, -72.710686),
            'VA': (37.769337, -78.169968),
            'WA': (47.400902, -120.659619),
            'WV': (38.491226, -80.954247),
            'WI': (44.268543, -89.616508),
            'WY': (42.755966, -107.302490),
        }

        # Créer la carte choroplèthe
        fig = px.choropleth(
            data_frame=df_camp_fire_state,
            locations='STATE',
            locationmode='USA-states',
            color='FIRE_COUNT',
            color_continuous_scale='YlOrRd',
            scope='usa',
            labels={'FIRE_COUNT': 'Nombre de feux'},
            title='Nombre de feux de forêt et localisation des campings',
        )

        # Ajouter des points pour le nombre de campings sur chaque État
        # Extraire les latitudes et longitudes des États
        latitudes = [state_coords[state][0] for state in df_camp_fire_state['STATE']]
        longitudes = [state_coords[state][1] for state in df_camp_fire_state['STATE']]

        # Ajouter des points sur la carte
        fig.add_trace(go.Scattergeo(
            lat=latitudes,  # Latitude des États
            lon=longitudes,  # Longitude des États
            text=df_camp_fire_state['COUNT'],  # Nombre de campings à afficher lors du survol
            mode='markers',
            marker=dict(
                size=df_camp_fire_state['COUNT']/25 ,  # Ajuste la taille des points
                color='green',
                opacity=0.6,
                line=dict(width=0.3, color='black'),
            ),
            name='Nombre de campings'  # Nom de la légende
        ))

        # Mettre à jour la mise en page pour inclure un fond et d'autres options
        fig.update_layout(
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='lightgrey',
                subunitcolor='blue',
            ),
        )

        # Afficher la figure
        st.plotly_chart(fig)

        #Isoler les feux liés à des feux de camps
        df_fires_camp = df[df['STAT_CAUSE_DESCR'] == 'Feux de camps']
        df_fires_camp_state = df_fires_camp.groupby(['STATE']).size().reset_index(name='FIRE_COUNT')
        df_filtered=df_camp.merge(df_fires_camp_state, on='STATE', how='left')
        df_filtered

        df_camp_fires=df_filtered.groupby('STATE')['FIRE_COUNT'].size().reset_index(name='COUNT')
        df_camp_fires = df_filtered.groupby('STATE').agg(
            COUNT=('STATE', 'size'),         # Compte le nombre de lignes
            FIRE_COUNT=('FIRE_COUNT', 'first')  # Somme de la colonne FIRE_COUNT
        ).reset_index()
        df_camp_fires

        # Créer un dictionnaire avec les coordonnées (latitude, longitude) de chaque État
        state_coords = {
            'AL': (32.806671, -86.791397),
            'AK': (61.370716, -152.404419),
            'AZ': (33.729759, -111.431221),
            'AR': (34.969704, -92.373123),
            'CA': (36.116203, -119.681564),
            'CO': (39.059811, -105.311104),
            'CT': (41.597782, -72.755371),
            'DE': (39.318523, -75.507141),
            'FL': (27.766279, -81.686783),
            'GA': (33.040619, -83.643074),
            'HI': (21.094318, -157.498337),
            'ID': (44.240459, -114.478828),
            'IL': (40.349457, -88.986137),
            'IN': (39.849426, -86.258278),
            'IA': (42.011539, -93.210526),
            'KS': (39.063946, -98.387207),
            'KY': (37.668140, -84.670067),
            'LA': (31.169546, -91.867805),
            'ME': (45.367584, -69.381927),
            'MD': (39.063946, -76.802101),
            'MA': (42.230171, -71.530106),
            'MI': (43.326618, -84.536095),
            'MN': (45.694454, -93.900192),
            'MS': (32.741646, -89.678696),
            'MO': (38.456085, -92.288368),
            'MT': (46.921925, -110.454353),
            'NE': (41.492537, -99.901810),
            'NV': (38.502503, -117.023060),
            'NH': (43.193852, -71.572395),
            'NJ': (40.298904, -74.521011),
            'NM': (34.840515, -106.248482),
            'NY': (42.165726, -74.948051),
            'NC': (35.630066, -79.806419),
            'ND': (47.528912, -99.784012),
            'OH': (40.388783, -82.764915),
            'OK': (35.565342, -96.928917),
            'OR': (43.933, -120.558),
            'PA': (40.590752, -77.209755),
            'RI': (41.680893, -71.511780),
            'SC': (33.856892, -80.945007),
            'SD': (44.299782, -99.438828),
            'TN': (35.747845, -86.692345),
            'TX': (31.054487, -97.563461),
            'UT': (40.150032, -111.862434),
            'VT': (44.045876, -72.710686),
            'VA': (37.769337, -78.169968),
            'WA': (47.400902, -120.659619),
            'WV': (38.491226, -80.954247),
            'WI': (44.268543, -89.616508),
            'WY': (42.755966, -107.302490),
        }

        # Créer la carte choroplèthe
        fig = px.choropleth(
            data_frame=df_camp_fires,
            locations='STATE',
            locationmode='USA-states',
            color='FIRE_COUNT',
            color_continuous_scale='YlOrRd',
            scope='usa',
            labels={'FIRE_COUNT': 'Nombre de feux'},
            title='Nombre de feux liés aux feux de camps et localisation des campings',
        )

        # Ajouter des points pour le nombre de campings sur chaque État
        # Extraire les latitudes et longitudes des États
        latitudes = [state_coords[state][0] for state in df_camp_fire_state['STATE']]
        longitudes = [state_coords[state][1] for state in df_camp_fire_state['STATE']]

        # Ajouter des points sur la carte
        fig.add_trace(go.Scattergeo(
            lat=latitudes,  # Latitude des États
            lon=longitudes,  # Longitude des États
            text=df_camp_fires['COUNT'],  # Nombre de campings à afficher lors du survol
            mode='markers',
            marker=dict(
                size=df_camp_fires['COUNT']/25 ,  # Ajuste la taille des points
                color='green',
                opacity=0.6,
                line=dict(width=0.3, color='black'),
            ),
            name='Nombre de campings'  # Nom de la légende
        ))

        # Mettre à jour la mise en page pour inclure un fond et d'autres options
        fig.update_layout(
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='lightgrey',
                subunitcolor='blue',
            ),
        )

        # Afficher la figure
        st.plotly_chart(fig)




























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


###################################################################VISUALISATION CARTE DES USA

    if option == "Visualisation carte des USA":
        
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

    df = pd.read_csv("MergedFinal.csv")
    st.dataframe(df.head(10))

if page == pages[3]:
    st.markdown('<h1 style="color: red;">MODELISATION</h1>', unsafe_allow_html=True)



















    #PRE-PROCESSING: Séparation du jeu d'entrainement et du jeu de test
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











if page == pages[4]:
    st.markdown('<h1 style="color: red;">OPTIMISATION</h1>', unsafe_allow_html=True)

if page == pages[5]:
    st.markdown('<h1 style="color: red;">CONCLUSION</h1>', unsafe_allow_html=True)
