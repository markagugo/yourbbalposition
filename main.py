import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import math
import pandas as pd
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split

data = pd.read_csv('NbaPlayersData.csv')

x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

pr = PolynomialFeatures(degree=8)
xp = pr.fit_transform(x_train)

ln_rg2 = LinearRegression()
ln_rg2.fit(xp, y_train)



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    height = request.form['height']
    weight = request.form['weight']
    features = (height, weight)
    c = [float(i) for i in features]
    print(c)
    # final_features = [np.array(c)]
    features = np.array([float(height), float(weight)]).reshape(1, -1)
    prediction = ln_rg2.predict(pr.fit_transform(features))

    pl = 'You Should Play >> '

    if prediction[0] >= 1 and prediction[0] < 2:
        pred = pl + 'Fowarder'
    elif prediction[0] >=2 and prediction[0] < 3:
        pred = pl + 'Center'
    elif prediction[0] >=3 and prediction[0] < 4:
        pred = pl + 'Center - Fowarder'
    elif prediction[0] >=4 and prediction[0] < 5:
        pred = pl + 'Guard'
    elif prediction[0] >=5 and prediction[0] < 6:
        pred = pl + 'Fowarder - Center'
    elif prediction[0] >=6 and prediction[0] < 7:
        pred = pl + 'Fowarder - Guard'
    elif prediction[0] >=7:
        pred = pl + 'Guard - Fowarder'

    return render_template('index.html', prediction_text=pred)

if __name__ == "__main__":
    app.run(debug=True)