from flask import Flask, jsonify, request, make_response
import pickle
import pandas as pd
import numpy as np

# cria o app flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

#importa modelo e scaler como objetos
model = pickle.load(open('./models/modelo_KNN.pkl', 'rb'))
# scaler = pickle.load(open('./model/std_scalar.pkl','rb'))

#labels para colunas do df
labels = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

@app.route('/',methods=['GET'])
def main():
    return 'main'

# decorator do flask para rotear caminho a função
@app.route('/predict',methods=['POST'])
def predict():
    features = request.get_json()

    for key in labels:
        if key not in features:
            return jsonify({'error': f'A chave "{key}" está faltando no JSON enviado.'})

    # df_scalar = []

    # for cols in features.keys():
    #     z_prod = (features[cols]-scaler[cols][0])/scaler[cols][1]
    #     df_scalar.append(z_prod)
    
    lista = []

    for value in features.values():
        lista.append(float(value))

    df = pd.DataFrame([lista], columns=labels)

    prediction = model.predict(df)

    return make_response(
        jsonify(mensagem = "Previsao feita com sucesso",
                dado = int(prediction[0])))


if __name__ == '__main__':
    app.run(debug=True)