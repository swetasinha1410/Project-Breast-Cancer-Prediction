import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
import pandas as pd
from flask import Flask, request, render_template


app = Flask(__name__)



@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST']) 
def prediction():
    if  request.method=='POST':
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        area_mean = float(request.form['area_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst= float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst= float(request.form['fractal_dimension_worst'])
        
        
        
        
        
        
        val = np.array([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,
        concave_points_mean,symmetry_mean,fractal_dimension_mean,
        radius_se, texture_se,perimeter_se,area_se,smoothness_se,
        compactness_se,concavity_se,concave_points_se,symmetry_se,
        fractal_dimension_se,radius_worst,texture_worst,
        perimeter_worst,area_worst,smoothness_worst,
        compactness_worst,concavity_worst,concave_points_worst,
        symmetry_worst, fractal_dimension_worst])
        
        features_name = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
        
        #input_features = [float(x) for x in request.form.values()]
        #features_value = [np.array(input_features)]
    
        model= joblib.load(open("Breast_cancer_v1.0.model","rb"))
        index=[1] #serial number
    
        my_data=pd.DataFrame([val],index)
        prediction =model.predict(my_data)
    
       
        if prediction == 0:
            res_val = "** breast cancer **"
        else:
            res_val = "no breast cancer"
        return render_template('index.html',prediction_text=res_val)
    return render_template('index.html')   

    
   # run application
if __name__ == "__main__":
    app.run(debug=True) 
    
    
      