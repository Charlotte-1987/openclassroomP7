import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import plotly.graph_objects as go
from urllib.request import urlopen
import shap
import joblib
from PIL import Image

shap.initjs()

st.set_page_config(
    page_title="Dashboard_Prêt à dépenser",
    layout="wide", # Use the full page instead of a narrow central column
    initial_sidebar_state="expanded",
    )

# st.set_option('deprecation.showPyplotGlobalUse', False) 
# https://medium.com/@data.science.enthusiast/create-web-apps-for-your-ml-model-using-python-and-streamlit-cc966142633d


######################
# importation des données
######################
PATH = "dashboard/DATA_DASHBOARD/"
df_sample_display = pd.read_csv(PATH+"df_sample_display.csv")
df_sample_modelling = pd.read_csv(PATH+"df_sample_modelling.csv")

model = joblib.load('lgbm.joblib')

######################
# fonctions
######################

# Fonction de visualisations des graphiques de comparaison d'un client dans le portefeuille

def plot_stats(feature, label_rotation=False, bins=False):

    fig, ax1 = plt.subplots(figsize=(10,5))
    
    if (bins):
        s = sns.histplot(ax=ax1, x = feature, hue="TARGET", data=df_sample_display, palette="rocket",stat="percent", multiple="dodge", bins = 500)
        ax1.set_xlim(0,200000)
    else:
        s = sns.histplot(ax=ax1, x = feature, hue="TARGET", data=df_sample_display, palette="rocket",stat="percent", multiple="dodge")
    
    
    ax1.set_title(feature+'\n', fontdict={'fontsize' : 15, 'fontweight' : 'bold'}) 
    ax1.set(ylabel = "Pourcentage de clients")
    
    ax1.axvline((char), color="blue", linestyle='--', label = 'Client')
    ax1.legend(['Client','Crédit en défaut','Crédit remboursé' ])
    
    if(label_rotation):
        #s.set_xticklabels(s.get_xticklabels(),rotation=90)
        ax1.tick_params(axis='x', rotation=70)
            
    st.pyplot(fig)









######################
# sidebar
######################

sb = st.sidebar

image = Image.open('logo_2.png')
sb.image(image, use_column_width='True')
sb.title("Prêt à dépenser")


display_mission = sb.checkbox("Descriptions de la mission")
if display_mission == True:
    sb.write("Cette application est un outil de scoring d'attribution de crédit. Elle permet également de visualiser les données des clients et leur positiionnement dans le portefeuille")

sb.title("Identifiant du client")
id_list = df_sample_display["ID DU CLIENT"].tolist()
id_client = sb.selectbox("identification client", id_list)

sb.title("Informations du portefeuille")
display_comparaison = sb.checkbox("Comparatif client")



######################
# Data client
###################### 
data_client = df_sample_modelling[df_sample_modelling['SK_ID_CURR']==id_client] # on identifie la ligne du client
raw = data_client.drop('SK_ID_CURR', axis=1) # on retire l'identifiant
temp_index = data_client.index.to_list() # on cherche l'index du client, sera utile pour le shap waterfall
index_client = int(temp_index[0])




######################
# main page 
######################

st.title('Prêt à dépenser - Etude de crédit')
st.subheader('Visualisation globale de la relation et du crédit demandé')

col1, col2 = st.columns([1, 1])

with col1:
    st.write("Données du client")
    df_sample_client = df_sample_display[['ID DU CLIENT','SEXE','AGE', 'STATUT MARITAL', 
                                    "NBRE D'ENFANTS",'EMPLOI', 'REVENU ANNUEL']].loc[df_sample_display['ID DU CLIENT'] == id_client]
    st.write(df_sample_client)


with col2:
    st.write("Données du crédit")
    df_sample_credit = df_sample_display[['TYPE DE CONTRAT','MONTANT DU CREDIT','ANNUITE DU CREDIT', 
                                    'RATIO ENDETTEMENT']].loc[df_sample_display['ID DU CLIENT'] == id_client]

    st.write(df_sample_credit)
    
    




st.subheader('Prédiction du crédit - Variables importantes')

col3, col4 = st.columns([1, 1])

if st.button('Prédiction du crédit'):
    with col3:
        data = json.dumps({'client':id_client})
        reponse = requests.post('http://127.0.0.1:5000/predict', data)
        proba_client = reponse.text
        proba_client2 = float(proba_client)*100
    
        st.write(f"La probabilité de rembourser le crédit: **{proba_client2} %**")

        # Jauge du scoring
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = proba_client2, # prédict_ proba
            mode = "gauge+number",
            title = {'text': "Scoring"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkslategrey"},
                    'steps' : [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 75], 'color': "darkorange"},
                        {'range': [75, 100], 'color': "yellowgreen"}],
                    'threshold' : {'line': {'color': "red", 'width': 3}, 'thickness': 0.5, 'value': 60}}))
        st.plotly_chart(fig)

        st.write(" Score rouge : crédit refusé")
        st.write(" Score orange : crédit à l'étude")
        st.write(" Score vert : crédit accepté")
 
    with col4:
        st.subheader('Importances des features')
        df_shap = df_sample_modelling.drop('SK_ID_CURR', axis=1)         
        explainer = shap.explainers.Tree(model)
        shap_values = explainer(df_shap)
        fig, ax = plt.subplots(figsize=(5,5))
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[index_client].values[:,0], feature_names=df_shap.columns)
        st.pyplot(fig)



    



######################
## INFORMATION PORTEFEUILLE
# Comparatif client
######################

# Dataframe utilisé pour les graphiques de visualisation
df_sample_client_vizu = df_sample_display[['ID DU CLIENT','TARGET','AGE', 'STATUT MARITAL', 
                            'EMPLOI', 'REVENU ANNUEL']].loc[df_sample_display['ID DU CLIENT'] == id_client]



if display_comparaison == True: # Bouton comparatif client
    col5, col6 = st.columns([1, 1])
    with col5:
        st.subheader('Comparaison avec le portefeuille')
        # col1.plotly_chart(histogram(df_train, x=num_plots[0], client=[df_test, input_client]), use_container_width=True)
        # st.plotly_chart(histogram(importantes_features_clients_test, use_container_width=True))
        feature = st.selectbox("Choix de la variable: ",
                        ['AGE', 'STATUT MARITAL', 'EMPLOI', 'REVENU ANNUEL'])
        
        if feature == 'AGE':
            char = df_sample_client_vizu.iloc[0,2]
            plot_stats('AGE')

        elif feature == 'STATUT MARITAL':
            char = df_sample_client_vizu.iloc[0,3]
            plot_stats('STATUT MARITAL')

        elif feature == 'EMPLOI':
            char = df_sample_client_vizu.iloc[0,4]
            plot_stats('EMPLOI', label_rotation=True)

        else:
            char = df_sample_client_vizu.iloc[0,5]
            plot_stats('REVENU ANNUEL', bins=True)

    with col6:
        st.subheader('Variables importantes')

        fig, ax = plt.subplots()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(raw)
        shap.summary_plot(shap_values, raw, max_display=10, plot_type ="bar", plot_size=(10, 10))
        st.pyplot(fig)




print("Tout est OK")
