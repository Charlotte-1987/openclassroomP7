from flask import Flask, jsonify, request
import pandas as pd
import joblib


app = Flask(__name__)



######################
# importation du modèle
######################
model = joblib.load('lgbm.joblib')


######################
# importation des données
######################
PATH = "dashboard/DATA_DASHBOARD/"
df_sample_modelling = pd.read_csv(PATH+"df_sample_modelling.csv")


######################
# fonction de prediction
######################
#id_client = 100291 #test

@app.route('/predict',methods=['POST'])
def prediction():
    data = request.get_json(force=True)
    print(data)
    id_client = data['client']
    data_client = df_sample_modelling[df_sample_modelling['SK_ID_CURR']==id_client] # on identifie la ligne du client
    raw = data_client.drop('SK_ID_CURR', axis=1) # on retire l'identifiant

    # prédiction du 0 donc du OK, puis transformé en float pour être lu dans la jauge
    proba_client = round(float(model.predict_proba(raw)[:,0]),2)
    print(proba_client)

    return jsonify(proba_client)




# if a request POST is made on this url, it runs the running function below
# https://littlebigcode.fr/modele-api-flask-web-service/



if __name__ == '__main__':
	app.run(debug=True) # supprimer cette option quand on le mettra en ligne
