import streamlit as st
import pickle
import numpy as np



#import the model 
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title
st.title("Laptop Price Predictor")

col1, col2, col3 = st.columns(3)

with col1:

    # brand
    company = st.selectbox('Brand', df['Company'].unique())

    #type
    types = st.selectbox('type', df['TypeName'].unique())

    #ram using slider 
    # ram = st.slider('Ram (GB)', 2,4,8.12,16,32,64)

    # custom_values = [2, 4, 8, 12, 16, 32, 64]
    # ram = st.slider('Ram (GB)', min_value=min(custom_values), max_value=max(custom_values), value=8, step=1, format='%d')


    #ram 
    ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 32, 64])

    #wieght
    weight = st.number_input('Weight of the laptop')


    #touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

    #ips
    ips = st.selectbox('IPS', ['No', 'Yes'])

    #screen size
    screen_size = st.number_input('Screen Size')

with col2:
    pass

with col3:
      #resolution
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2304x1440'])


    #cpu
    cpu = st.selectbox('CPU', df['Cpu name'].unique())

    #hdd
    hdd = st.selectbox('HDD ( in GB)', [0, 128, 256, 512, 1024, 2048])


    #ssd
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])


    #gpu
    gpu = st.selectbox('GPU', df['gpu brand'].unique())


    #os
    os = st.selectbox('OS', df['os'].unique())

  



if st.button("Predict Price"):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:   
        touchscreen = 0


    if ips == 'Yes':
        ips = 1
    else:
        ips = 0


    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, types, ram, weight, touchscreen, ips, ppi, cpu, hdd , ssd, gpu, os]).T
  

    query = query.reshape(1,12)
    st.title("The predicted price of your laptop is: " + str(int(np.exp(pipe.predict(query)[0]))))


#   #apply one hot encoding
#   query = pd.get_dummies(query)
  
#   #remove additional columns
#   query = query.reindex(columns=model_columns, fill_value=0)
  
#   #prediction
#   y_pred = pipe.predict(query)
#   st.title("The predicted price of this configuration is {}".format(y_pred[0])+" Euros
 