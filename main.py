import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    """
    Predict the class of a plant disease based on the uploaded image.
    """
    model = tf.keras.models.load_model("Plant_Disease_Dataset/trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE DETECTOR")
    image_path = "Plant_Disease_Dataset/train\Apple___Apple_scab/0d3c0790-7833-470b-ac6e-94d0a3bf3e7c___FREC_Scab 2959_90deg.JPG"
    st.image(image_path, use_column_width=True)
    st.markdown(
        """
        PLANT DISEASE DETECTOR 
        ### Features
        - **Upload and Analyze:** Simply upload an image, and the system will do the rest.
        - **Comprehensive Detection:** Capable of recognizing a wide variety of plant diseases.
        - **Fast and Reliable:** Get results instantly with high accuracy.

        ### Diseases We Can Detect
        - Apple: **Apple scab, Black rot, Cedar apple rust, Healthy**
        - Blueberry: **Healthy**
        - Cherry: **Powdery mildew, Healthy**
        - Corn: **Cercospora leaf spot, Common rust, Northern leaf blight, Healthy**
        - Grape: **Black rot, Esca (Black Measles), Leaf blight, Healthy**
        - Orange: **Citrus greening (Haunglongbing)**
        - Peach: **Bacterial spot, Healthy**
        - Pepper: **Bacterial spot, Healthy**
        - Potato: **Early blight, Late blight, Healthy**
        - Raspberry: **Healthy**
        - Soybean: **Healthy**
        - Squash: **Powdery mildew**
        - Strawberry: **Leaf scorch, Healthy**
        - Tomato: **Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy**

        ### Get Started
        Navigate to the **Disease Recognition** page in the sidebar to begin your analysis.
        """
    )

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            with st.spinner("Analyzing the image. Please wait..."):
                result_index = model_prediction(test_image)

            # Define class names
            class_names = [
                'Apple - Apple scab', 'Apple - Black rot', 'Apple - Cedar apple rust', 'Apple - Healthy',
                'Blueberry - Healthy', 'Cherry - Powdery mildew', 'Cherry - Healthy',
                'Corn - Cercospora leaf spot (Gray leaf spot)', 'Corn - Common rust',
                'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black rot',
                'Grape - Esca (Black Measles)', 'Grape - Leaf blight (Isariopsis Leaf Spot)', 'Grape - Healthy',
                'Orange - Citrus greening (Haunglongbing)', 'Peach - Bacterial spot', 'Peach - Healthy',
                'Pepper - Bacterial spot', 'Pepper - Healthy', 'Potato - Early blight', 'Potato - Late blight',
                'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery mildew',
                'Strawberry - Leaf scorch', 'Strawberry - Healthy', 'Tomato - Bacterial spot',
                'Tomato - Early blight', 'Tomato - Late blight', 'Tomato - Leaf Mold',
                'Tomato - Septoria leaf spot', 'Tomato - Spider mites (Two-spotted spider mite)',
                'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
            ]

            st.success(f"Prediction: {class_names[result_index]}")
