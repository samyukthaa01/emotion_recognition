# Import the necessary libraries for model loading and JSON parsing
import tensorflow as tf
import json

# Load the JSON file containing the model architecture
with open('model.json', 'r') as json_file:
    model_json = json.load(json_file)

# Reconstruct the model from the JSON
model = tf.keras.models.model_from_json(model_json)

# Load the model weights
model.load_weights('training_modelw.h5')

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
 
# give a title to our app
st.title("Emotion Recognition Application")
st.header("Choose an Option")
# Add a radio button to choose between CNN and KNN
method = st.sidebar.radio("Choose Emotion Recognition Method", ("CNN", "KNN"))
# Add a radio button to choose between webcam and image upload
input_type = st.sidebar.radio("Choose Input Type", ("Webcam", "Upload Image"))
# Based on user selections, perform the chosen action
if method == "CNN":
    st.write("You selected Emotion Recognition with CNN.")
# Within the Streamlit code, when you handle image upload and perform inference:
if method == "CNN" and input_type == "Upload Image" and uploaded_image is not None:
    st.write("You selected Emotion Recognition with CNN.")
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    processed_image = preprocess_image(uploaded_image)
    prediction = model.predict(processed_image)

    # Display the prediction results
    st.write("Emotion Prediction:")
    # Depending on your model's output format, you may need to interpret the prediction.
    # For example, if it's a classification task, you can display class labels and probabilities.
    st.write(prediction)
elif method == "KNN":
    st.write("You selected Emotion Recognition with KNN.")
# You can add code here to run KNN-based emotion recognition.
if input_type == "Webcam":
     st.write("You selected Webcam Input.")
# You can add code here to capture webcam input.
elif input_type == "Upload Image":
     st.write("You selected Image Uploading.")
     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])






