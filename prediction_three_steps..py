# Import the libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use('fivethirtyeight')
from keras.models import  load_model

import warnings
warnings.filterwarnings("ignore")

import yfinance as fyf
fyf.pdr_override() # <-- Here is the fix

from datetime import datetime
start_date = datetime(2012,1,1)
end_date = datetime(2019,12,17)

#Get the stock quote
class CustomMinMaxScaler:
    def __init__(self, minimum, maximum, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_value = minimum
        self.max_value = maximum


    def transform(self, data):

        scaled_data = (data - self.min_value) / (self.max_value - self.min_value)
        scaled_data = scaled_data * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data

    def inverse_transform(self, scaled_data):

        unscaled_data = (scaled_data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        unscaled_data = unscaled_data * (self.max_value - self.min_value) + self.min_value
        return unscaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    


def scaler_google():    
    scaler = CustomMinMaxScaler(13.990240097045898, 68.03500366210938,feature_range=(0, 1))
    return scaler

def scaler_apple():
    scaler = CustomMinMaxScaler(13.947500228881836, 69.96499633789062,feature_range=(0, 1))
    return scaler


# Initializing Scalers
google_scaler =  scaler_google()
apple_scaler = scaler_apple()


# Load the trained models
google_model = load_model("C:/Users/Shenal Harshana/.spyder-py3/autosave/google_best.h5")
apple_model  = load_model('C:/Users/Shenal Harshana/.spyder-py3/autosave/apple_best.h5')




def y_prediction_three_steps(model,input_data,scaler_close):
  input_scaled =[google_scaler.transform(np.array(i).reshape(1,-1)) for i in input_data]
  input_ = [i.item() for i in input_scaled]

  y_pre_3 = []
  
  try :
      try:
        pred1 = model.predict([input_])
        x_test1 = input_[1:]
        x_test1.append(pred1[0][0])
    
        pred2 = model.predict([x_test1])
        x_test2 = x_test1[1:]
        x_test2.append(pred2[0][0])
    
        pred3 = model.predict([x_test2])
        x_test3 = x_test2[1:]
        x_test3.append(pred3[0][0])
    
        y_pre_3.append(pred1[0][0])
        y_pre_3.append(pred2[0][0])
        y_pre_3.append(pred3[0][0])
    
        return y_pre_3
    
      except:
        pred1 = model.predict(np.array([input_] ).reshape(1,-1))
        x_test1 = input_[1:]
        x_test1.append(pred1[0])
    
        pred2 = model.predict(np.array(x_test1).reshape(1,-1))
        x_test2 = x_test1[1:]
        x_test2.append(pred2[0])
    
        pred3 = model.predict(np.array([x_test2]).reshape(1,-1))
        x_test3 = x_test2[1:]
        x_test3.append(pred3[0])
    
        y_pre_3.append(pred1[0])
        y_pre_3.append(pred2[0])
        y_pre_3.append(pred3[0])
    
        return y_pre_3
  except:
    pred1 = model.predict(input_)
    x_test1 = input_[1:]
    x_test1.append(pred1[0][0])

    pred2 = model.predict([x_test1])
    x_test2 = x_test1[1:]
    x_test2.append(pred2[0][0])

    pred3 = model.predict([x_test2])
    x_test3 = x_test2[1:]
    x_test3.append(pred3[0][0])
    
    y_pre_3.append(pred1[0][0])
    y_pre_3.append(pred2[0][0])
    y_pre_3.append(pred3[0][0])
    
  return y_pre_3

# Converting to original values of prediction (inverse transform)
def original_values_of_predictions(y_prediction_three_steps,scaler_close):

  original_values = [scaler_close.inverse_transform(i) for i in y_prediction_three_steps]
  
  return original_values




st.title('Stock Price prediction')

st.sidebar.markdown("<div class='sidebar-header'>Input Stock Prices and Stock Selection</div>", unsafe_allow_html=True)

selected_model = st.sidebar.selectbox("Select The Stock", ["Google", "Apple"], key="stock_selector")

# Sidebar Input for previous 7 days' stock values
previous_days = []

for i in range(7):
    day = st.sidebar.number_input(f"{7-i} Day before Stock Price", step=0.01, key=f"stock_price_{i}")
    previous_days.append(day)

# Ensure that there are 7 values entered for previous days
if len(previous_days) != 7:
    st.sidebar.warning("Please enter exactly 7 values for previous days' stock.")
else:
    if selected_model == "Google":
        original_list1  = original_values_of_predictions(y_prediction_three_steps(google_model ,previous_days,google_scaler ),google_scaler )
        predicted_prices = np.array(original_list1).flatten()
        numbers = [i for i in previous_days]+[i for i in predicted_prices]
        col_names = ['7 day beofore','6 day beofore','5 day beofore','4 day beofore','3 day beofore','2 day beofore','1 day beofore','Today','Tommorow','Day After Tommorow']
        data_frame_pred = pd.DataFrame([numbers], columns=col_names)
        
    elif selected_model == "Apple":
        original_list2  = original_values_of_predictions(y_prediction_three_steps(apple_model ,previous_days,apple_scaler ),apple_scaler )
        predicted_prices = np.array(original_list2).flatten()
        numbers = [i for i in previous_days]+[i for i in predicted_prices]
        col_names = ['7 day beofore','6 day beofore','5 day beofore','4 day beofore','3 day beofore','2 day beofore','1 day beofore','Today','Tommorow','Day After Tommorow']
        data_frame_pred = pd.DataFrame([numbers], columns=col_names)
        

      


    if st.sidebar.button('Predict Values') ==1 :
        fig,ax = plt.subplots()
        days_ahead = [i for i in range(1, 11)]  # Adjusted for displaying previous values
        ax.plot(days_ahead[:7], previous_days, marker='o', color='red', label='Previous Days')
        ax.plot(days_ahead[7:], predicted_prices, marker='o', color='green', label='Predicted Price')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Stock Price')
        ax.set_title('Stock Price Prediction for '+str(selected_model))
        ax.legend()
        # Display the plot using Streamlit
        st.pyplot(fig)
        st.dataframe(data_frame_pred)
    else:
        pass

