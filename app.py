import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

dataset=pd.read_csv('outliers_data.csv')

from sklearn.preprocessing import MinMaxScaler


columns_to_normalize = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
    'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude',
    'Weatherconditions', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
    'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City', 
    'Food_Preparation_time_minutes', 'distance', 'day', 'month', 'quarter', 'year',
    'day_of_week', 'is_month_start', 'is_month_end', 'is_quarter_start',
    'is_quarter_end', 'is_year_start', 'is_year_end', 'is_weekend'
]


scaler = MinMaxScaler()


normalized_data = scaler.fit_transform(dataset[columns_to_normalize])


normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Weatherconditions', 'Road_traffic_density',
            'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries',
            'Festival', 'City', 'Food_Preparation_time_minutes', 'distance', 'is_weekend']
target = 'Time_taken(min)'


X = normalized_df[features]
y = dataset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=15,max_depth=10, min_samples_leaf=5, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

#from PIL import Image
#with st.container():
 #   image = Image.open('vecteezy_delivery-man-on-scooter_.jpg')
  #  st.image(image)

predictio_n, analytic_s = st.tabs(["Time Prediction", "Analytics"])




with predictio_n:
   
    image = Image.open('black.jpg')
    st.image(image)

    values_age = st.sidebar.slider(
        'Age',
        18,40,30)
    #st.write('Values:', values_age)

    values_ratings = st.sidebar.slider(
        'Ratings',
        1.0,5.0,4.0)
    #st.write('Values:', values_ratings)

    festival_mapping = {'No': 0, 'Yes':1}

    option6 = st.sidebar.selectbox(
        'festival',
        festival_mapping.keys())

    selected_festival=festival_mapping[option6]

    values_multiple_deliveries = st.sidebar.slider(
        'Multiple deliveries',
        0,1,0)
    
    values_weekend = st.sidebar.slider(
        'Weekend',
        0,1)

    values_distance=st.sidebar.slider(
        'Distance b/w Restaurent and location',
        1,20,1)

    values_prep_time=st.sidebar.slider(
        'Preperation time',
        5,15)

    values_vehicle_cond=st.sidebar.slider(
        'vehicle condition',
        0,2)




    map_order={'Snack':3,'Meal':2,'Drinks':1,'Buffet':0}

    #column_map_order=['Type','encoded_snack']
    #item=[]
    #for i in map_order.items:

    option1 = st.selectbox(
        'Select Type of Order',
        map_order.keys())

    selected_meal=map_order[option1]

    #st.write('You selected:', selected_meal)



    map_weather={'conditions Fog':1,'conditions Stormy':3,'conditions Cloudy':0,'conditions Sandstorms':2,'conditions Windy':5,'conditions Sunny':4}


    option2 = st.selectbox(
        'Select Weather Condition',
        map_weather.keys())

    selected_weather=map_weather[option2]

    #st.write('You selected:', selected_weather)

    vehicle_mapping = {'motorcycle': 1, 'scooter': 2, 'electric_scooter':0}

    option3 = st.selectbox(
        'Select Vehicle Type',
        vehicle_mapping.keys())

    selected_vehicle=vehicle_mapping[option3]

    map_density={'Low':2,'Jam':1,'Medium':3,'High':0}

    option4 = st.selectbox(
        'Select road density',
        map_density.keys())

    selected_density=map_density[option4]

    city_mapping = {'Metropolitan':0,'Urban':2,'town': 3,'Semi-Urban':1}

    option5 = st.selectbox(
        'Select City type',
        city_mapping.keys())

    selected_city=city_mapping[option5]




    new_data = {
        'Delivery_person_Age': [values_age], 
        'Delivery_person_Ratings': [values_ratings],
        'Weatherconditions': [selected_weather],  
        'Road_traffic_density': [selected_density],  
        'Vehicle_condition': [values_vehicle_cond],  
        'Type_of_order': [selected_meal], 
        'Type_of_vehicle': [selected_vehicle],  
        'multiple_deliveries':[values_multiple_deliveries],  
        'Festival': [selected_festival], 
        'City': [selected_city],  
        'Food_Preparation_time_minutes': [values_prep_time],
        'distance': [values_distance],
        'is_weekend': [values_weekend],  
    }


    new_data_df = pd.DataFrame(new_data)


    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)

    predicted_time_taken = rf_model.predict(new_data_df)
    st.write('Time predicted', f"{predicted_time_taken[0]:.2f} min")
    time.sleep(1)
    my_bar.empty()
    st.success('This is a success message!', icon="✅")

    st.button("run")



    #predicted_time_taken = rf_model.predict(new_data_df)


    #print(f"Predicted Time Taken (min): {predicted_time_taken[0]:.2f}")
    #st.write('Time predicted', f"{predicted_time_taken[0]:.2f} min")

with analytic_s:
    st.header("Analytics")
    sns.set(style="darkgrid") 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Delivery_person_Age', data=dataset,ax=ax)
    st.pyplot(fig)

    sns.set(style="darkgrid") 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Delivery_person_Ratings', data=dataset,ax=ax)
    st.pyplot(fig)
    st.header("Analytics")
    
    st.write('Bar Plot of Road Traffic Density vs. Time Taken')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Road_traffic_density', y='Time_taken(min)', data=dataset, ax=ax)
    ax.set_xticklabels(['Low', 'Jam', 'Medium', 'High'])
    st.pyplot(fig)

    st.write('City vs. Time Taken')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='City', y='Time_taken(min)', data=dataset, ax=ax)
    ax.set_xticklabels(['Metropolitan','Urban','town','Semi-Urban'])
    st.pyplot(fig)


    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Delivery_person_Age', y='Time_taken(min)', data=dataset, hue='distance', palette='viridis', ax=ax)
    sns.regplot(x='Delivery_person_Age', y='Time_taken(min)', data=dataset, scatter=False, ax=ax)
    st.pyplot(fig)

    fig = px.scatter(dataset, x='Delivery_person_Age', y='Time_taken(min)', color='distance', trendline='ols')
    st.plotly_chart(fig)

    fig = px.scatter(dataset, x='Delivery_person_Ratings', y='Time_taken(min)', color='distance', trendline='ols')
    st.plotly_chart(fig)

    fig = px.scatter(dataset, x='distance', y='Time_taken(min)', color='distance', trendline='ols')
    st.plotly_chart(fig)

    
    dataset['Order_Date'] = pd.to_datetime(dataset['Order_Date'])
    count_df = dataset.groupby(dataset['Order_Date'].dt.to_period("M")).size().reset_index(name='count')
    count_df['Order_Date'] = count_df['Order_Date'].dt.strftime('%Y-%m')
    fig = px.line(count_df, x='Order_Date', y='count', title='Time Series Plot', markers=True)
    st.plotly_chart(fig)


    sns.set(style="darkgrid") 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='is_weekend', data=dataset,ax=ax)
    ax.set_xticklabels(['Yes', 'No'])
    st.pyplot(fig)

    sns.set(style="darkgrid") 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Festival', data=dataset,ax=ax)
    ax.set_xticklabels(['Yes', 'No'])
    st.pyplot(fig)


    sns.set(style="darkgrid") 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='multiple_deliveries', data=dataset,ax=ax)
    ax.set_xticklabels(['1', '2','3','4'])
    st.pyplot(fig)








    
   
        


   



