# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:52:37 2020

@author: Faizan Akhtar123
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__) 

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text=f'Salary of the Employee is {output}$')


if __name__ == "__main__":
    app.run(debug=True)
    