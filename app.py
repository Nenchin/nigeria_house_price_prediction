# importing needed libraries
import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Nigeria House Price Prediction",
                   page_icon="🧊")

# Custom CSS for background image
page_bg_img = '''
<style>
body {
background-image: url('unnamed-1.jpg');
background-size: cover;
background-color: green;
}
</style>
'''

st.html(page_bg_img)

st.title("Nigeria House Price Predictor")


c1, c2 = st.columns([0.7,0.3], vertical_alignment="center")

def input_features():
    col1, col2 = c1.columns(2)
    bedrooms = col1.number_input("Bedrooms", 
                                value=1, step=1,
                                  placeholder="Number of bedrooms")
    
    #bathrooms = col2.number_input("Bathrooms", 
    #                         value="min", min_value=0, step=1,
    #                          placeholder=" Number of bathrooms")
    
    title = col1.radio("House type", [1,2,3,4,5,6,7], 
                    format_func=lambda x: ["Block of Flats", 
                                           "Detached Bungalow",
                                           "Detached Duplex",
                                           "Semi Detached Bungalow",
                                           "Semi Detached Duplex", 
                                           "Terraced Bungalow",
                                           "Terraced Duplexes"][x-1])
    
    parking_space = col2.number_input("Parking space",
                           value="min", min_value=0,
                              placeholder="Number of cars parking spaces can contain")
        
    lat = col2.number_input("Latitude", value=0.001, min_value=0.001,
                            step=0.001, placeholder="Latitude")

    lon = col2.number_input("Longitude", value=0.001, min_value=0.001,
                            step=0.001, placeholder="Longitude")
    
    
    data = {
            "bedrooms":bedrooms,
            #"bathrooms":bathrooms,
            "parking_space":parking_space,
            "title":title,
            "lat":lat,
            "lon":lon
            }
    features_df = pd.DataFrame(data, index=[0], columns=["bedrooms", "parking_space", "title", "lat", "lon"])
    
    c1.write(features_df)
    
    return features_df

input_df = input_features()

scaler = jb.load(r"C:\Users\Nenchi\Documents\Python_Scripts\houseprice9ja\scaler1.pkl")
xgb_clf = jb.load(r"C:\Users\Nenchi\Documents\Python_Scripts\houseprice9ja\model.pkl")

#c2.image('unnamed-1.jpg')

if c1.button("Predict Input Data"):
    input_transf = scaler.transform(input_df)
    prediction = np.exp(xgb_clf.predict(input_transf))
    c2.write(prediction)
    

    
    c1.write(" # Predicted result")
    c1.write(pd.concat([input_df, pd.DataFrame(prediction,
                          columns=["result"],
                          index=[0])], axis=1))
    c1.write("---")
