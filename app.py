from flask import Flask, jsonify, render_template, request, make_response
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
    return render_template('index.html')

# decorator do flask para rotear caminho a função
@app.route('/predict_json',methods=['POST'])
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

@app.route('/predict', methods=['POST'])
def predict_index():
    try:
        gender = float(request.form.get('gender'))
    except (ValueError, TypeError):
        gender = 0

    try:
        age = float(request.form.get('age'))
    except (ValueError, TypeError):
        age = 0

    try:
        hypertension = float(request.form.get('hypertension'))
    except (ValueError, TypeError):
        hypertension = 0

    try:
        heart_disease = float(request.form.get('heart_disease'))
    except (ValueError, TypeError):
        heart_disease = 0

    try:
        ever_married = float(request.form.get('ever_married'))
    except (ValueError, TypeError):
        ever_married = 0

    try:
        work_type = float(request.form.get('work_type'))
    except (ValueError, TypeError):
        work_type = 0

    try:
        Residence_type = float(request.form.get('Residence_type'))
    except (ValueError, TypeError):
        Residence_type = 0

    try:
        avg_glucose_level = float(request.form.get('avg_glucose_level'))
    except (ValueError, TypeError):
        avg_glucose_level = 0

    try:
        bmi = float(request.form.get('bmi'))
    except (ValueError, TypeError):
        bmi = 0

    try:
        smoking_status = float(request.form.get('smoking_status'))
    except (ValueError, TypeError):
        smoking_status = 0

    features = pd.DataFrame(np.array([gender, age, hypertension, heart_disease, ever_married,
       work_type, Residence_type, avg_glucose_level, bmi,
       smoking_status]).reshape(1,10), columns=labels)
    
    # df_scalar = []

    # for cols in features.keys():
    #     z_prod = (features[cols]-scaler[cols][0])/scaler[cols][1]
    #     df_scalar.append(z_prod)

    # df_scalar = pd.DataFrame([df_scalar], columns=labels)
    prediction = model.predict(features)[0]

    if prediction==1:
        prediction = 'Yes'
    elif prediction == 0:
        prediction = 'No'

    print(prediction)
    
    return render_template('index.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)