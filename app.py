# import the streamlit library
import streamlit as st
 
# give a title to our app
st.title("Emotion Recognition Application")
st.sidebar.header("Choose an Option")
# Add a radio button to choose between CNN and KNN
method = st.sidebar.radio("Choose Emotion Recognition Method", ("CNN", "KNN"))
# Add a radio button to choose between webcam and image upload
input_type = st.sidebar.radio("Choose Input Type", ("Webcam", "Upload Image"))



