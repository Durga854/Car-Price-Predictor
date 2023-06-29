import streamlit as st
import pickle
import pandas as pd
import numpy as np

model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

class predict():
    def __init__(self):
        st.title("Car Price Predictor")
        st.markdown("Welcome to Car Price Predictor")
        companies=sorted(car['company'].unique())
        companies.insert(0,'Select Company')
        self.company=st.selectbox("Enter the car company:",companies)
        car_names=np.array(car[car['company']==self.company]['name'].tolist())
        car_names=np.unique(car_names)
        self.car_model=st.selectbox("Enter the car model:",sorted(car_names))
        self.year=st.selectbox("Enter the car manufacture year:",sorted(car['year'].unique(),reverse=True))
        self.fuel_type=st.selectbox("Enter the car fuel type:",car['fuel_type'].unique())
        self.driven=st.slider("Select the kilometers driven by car",0,999999)
        if(st.button("Submit")):
            self.prediction=float(model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=np.array([self.car_model,self.company,self.year,self.driven,self.fuel_type]).reshape(1, 5))))
            st.success(self.prediction)
    
if __name__=='__main__':
    predict()
