from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application= Flask(__name__)

app=application
# route for home page
@app.route('/')
def index():
    return render_template('index.html')

#route for predict page
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
        Gender=request.form.get('Gender'),
        Car_Owner=request.form.get('Car_Owner'),
        Property_Owner=request.form.get('Property_Owner'),
        Children=request.form.get('Children'),
        Annual_Income=request.form.get('Annual_Income'),
        Type_Income=request.form.get('Type_Income'),
        Education=request.form.get('Education'),
        Marital_Status=request.form.get('Marital_Status'),
        Housing_Type=request.form.get('Housing_Type'),
        Mobile_Phone=request.form.get('Mobile_Phone'),
        Work_Phone=request.form.get('Work_Phone'),
        Phone=request.form.get('Phone'),
        Email_Id=request.form.get('Email_Id'),
        Type_Occupation=request.form.get('Type_Occupation'),
        Family_Members=request.form.get('Family_Members'),
        Year_Of_Experience=request.form.get('Year_Of_Experience'),
        Age=request.form.get('Age')
        )

    pred_df=data.get_data_as_dataframe()
    print(pred_df)

    predict_pipeline=PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000)