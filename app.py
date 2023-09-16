# import the streamlit library
import streamlit as st
 
# give a title to our app
st.title("Emotion Recognition Application")
st.sidebar.header("Choose an Option")
# Add a radio button to choose between CNN and KNN
method = st.sidebar.radio("Choose Emotion Recognition Method", ("CNN", "KNN"))
# Add a radio button to choose between webcam and image upload
input_type = st.sidebar.radio("Choose Input Type", ("Webcam", "Upload Image"))
# Based on user selections, perform the chosen action
if method == "CNN":
    st.write("You selected Emotion Recognition with CNN.")
# You can add code here to run CNN-based emotion recognition.
elif method == "KNN":
    st.write("You selected Emotion Recognition with KNN.")
# You can add code here to run KNN-based emotion recognition.
if input_type == "Webcam":
     st.write("You selected Webcam Input.")
# You can add code here to capture webcam input.





