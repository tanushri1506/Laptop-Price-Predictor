import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand',df['Company'].unique())

type_name = st.selectbox('Type',df['TypeName'].unique())

ram = st.selectbox('RAM (in GB)',[2,4,6,8,12,16,24,32,64])

weight = st.number_input('Weight')

touch = st.selectbox('TouchScreen',['No','Yes'])

ips = st.selectbox('IPS',['No','Yes'])

screen_size = st.slider('Screen Size',10.0,18.0,13.0)

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU',df['CPU'].unique())

hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['GPU'].unique())

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    try:
        touch = 1 if touch == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        
        xres = int(resolution.split('x')[0])
        yres = int(resolution.split('x')[1])
        ppi = ((xres**2)+(yres**2))**0.5/screen_size

        query = np.array([
            company,type_name,ram,weight,touch,ips,ppi,cpu,hdd,ssd,gpu,os
            ],dtype=object)

        query = query.reshape(1,12)

        predicted_price = (np.exp(pipe.predict(query)[0]))

        st.title(f"Predicted Price for this description is ₹{int(predicted_price)}")
    
    except Exception as e:
        st.error(f"An Error Occured: {e}")