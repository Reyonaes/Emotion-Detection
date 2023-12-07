import streamlit as st
import cv2
from tensorflow import keras
import numpy as np
st.title("Emotion Detection")
st.write("Detects 7 emotions")
uploaded_img = st.file_uploader("Upload an image to detect emotion", type= ['png', 'jpg', 'jpeg'])
model=keras.models.load_model("Emotion_Model2.h5")
emotions=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
if uploaded_img is not None:
    with open("img.jpg",'wb') as f:
        f.write(uploaded_img.read())
    st.image("img.jpg")
    img=cv2.imread("img.jpg",0)
    img=cv2.resize(img, (48,48))
    img=np.array(img)
    img=img.reshape((48,48,1))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img=img.astype("float32")/255.0
    prob=model.predict(img)
    ind=np.argmax(prob)
    st.write("The emotion displayed is ",emotions[ind])
