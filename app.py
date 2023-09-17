import numpy as np
from PIL import Image
import tensorflow as tf
import json
import keras
import os

# Function to load and inspect model JSON
def load_and_inspect_model_json():
    # Load the JSON file containing the model architecture
    model_json_path = "C:\\Users\\user\\Downloads\\model.json"  # Replace with the correct path
    model_weights_path = "C:\\Users\\user\\Downloads\\training_modelw.h5"  # Replace with the correct path
    print("Model JSON path:", model_json_path)
    print("Model weights path:", model_weights_path)

    if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
        with open(model_json_path, 'r') as json_file:
            model_json = json.load(json_file)

        try:
            # Attempt to reconstruct the model from the JSON
            model = tf.keras.models.model_from_json(model_json)
            # Load the model weights
            model.load_weights(model_weights_path)

            # Print the content of model_json
            print("Model JSON content:", model_json)

        except Exception as e:
            # Handle any exceptions or errors that might occur during model loading
            print("Error loading the model:", str(e))

    else:
        print("Model files not found. Please check the file paths.")
        
# Define a mapping of class labels to human-readable emotions
class_emotions = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# Now, you have your model loaded and ready for inference
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))  # Adjust to match your model's input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values (assuming your model expects values in [0, 1])
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# import the streamlit library
import streamlit as st

# Load the model
model = load_and_inspect_model_json()

if model is not None:
    # give a title to our app
    st.title("Emotion Recognition Application")
    st.header("Choose an Option")
    # Add a radio button to choose between CNN and KNN
    method = st.sidebar.radio("Choose Emotion Recognition Method", ("CNN", "KNN"))
    # Add a radio button to choose between webcam and image upload
    input_type = st.sidebar.radio("Choose Input Type", ("Webcam", "Upload Image"))

    # Define a variable to store the uploaded image
    uploaded_image = None  # Initialize to None

    # Display the file upload button for image selection
    if input_type == "Upload Image":
        st.write("You selected Image Uploading.")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Add a submit button to trigger CNN inference
    if st.button("Submit"):
       if method == "CNN" and uploaded_image is not None:
           st.write("You selected Emotion Recognition with CNN.")
           st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
           processed_image = preprocess_image(uploaded_image)
           prediction = model.predict(processed_image)

        # Assuming your model predicts class labels (e.g., 0 for Angry, 1 for Happy, etc.)
           predicted_class = np.argmax(prediction)  # Get the index of the predicted class
           predicted_emotion = class_emotions.get(predicted_class, "Unknown")

        # Display the prediction result as text
           st.write("Predicted Emotion:", predicted_emotion)
        elif method == "KNN":
           st.write("You selected Emotion Recognition with KNN.")
            # You can add code here to run KNN-based emotion recognition.
        elif input_type == "Webcam":
           st.write("You selected Webcam Input.")
            # You can add code here to capture webcam input.

    # Optionally, you can include a message to instruct the user to click the submit button.
    st.write("Click the 'Submit' button to perform the selected action.")
else:
    st.error("Model failed to load. Please check the file paths or model format.")






