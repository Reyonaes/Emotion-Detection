import streamlit as st
from opencv-python import cv2
from tensorflow import keras
uploaded_img = st.file_uploader("Upload an image to detect emotion", type= ['png', 'jpg', 'jpeg'])
model=keras.models.load_model("Emotion_Model.h5")
emotions=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
if uploaded_img is not None:
    with open("img.jpg",'wb') as f:
        f.write(uploaded_img.read())
img=cv2.imread("img.jpg",0)
img=np.array(img)
img=img.reshape(img.shape[0],48,48,1)
ind=model.predict(img)
st.write("The emotion displayed is ",emotions[ind])
