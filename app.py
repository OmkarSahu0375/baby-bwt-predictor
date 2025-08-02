from flask import Flask, request, render_template
import pandas as pd
import pickle, os

app = Flask(__name__)


@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')




def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = float(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    cleaned_data = {
        'gestation': [gestation],
        'parity': [parity],
        'age': [age],
        'height': [height],
        'weight': [weight],
        'smoke': [smoke]
    }
    return cleaned_data
    
    



@app.route('/predict', methods = ["POST"])
def get_prediction():
    #get data from user in json format
    #baby_data = request.get_json()
    #get user data from the form when the user hit predict button
    baby_form_data = request.form

    # baby_form_data gives me all the data in string, so it is mandatory to clean it first.
    baby_form_data_clean = get_cleaned_data(baby_form_data)
    
    #converting the user data in Df
    baby_data_df = pd.DataFrame(baby_form_data_clean)

    #loading the machine learning model
    with open('model.pkl', mode='rb') as obj:
        model = pickle.load(obj)

    # making prediction 
    prediction = model.predict(baby_data_df)
    prediction_value = round(float(prediction[0]), 2)

    
    # return response in json format
    #response = {'Prediction': prediction}

    return render_template('index.html', prediction = prediction_value)


if __name__ == '__main__':
    app.run(debug = True)

