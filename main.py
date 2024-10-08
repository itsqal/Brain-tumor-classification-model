import model_load
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

model = model_load.model
transform = model_load.transform
model.eval()

# Streamlit app interface
st.title("Brain Tumor Detection")
st.markdown("""
This application uses a pre-trained model to classify brain MRI images 
as either **healthy** or **tumor present**. 
Upload an image to get the prediction!
""")

# File uploader
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image and make predictions
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)

    # Display the result
    if predicted.item() == 0:
        st.success("Result: **Healthy Brain**")
    else:
        st.error("Result: **Brain with Tumor**")

# Add more features or information if necessary
st.markdown("""
### About the Model
This model was trained on a dataset of brain MRI images and utilizes the ResNet50 architecture. 
It can classify images into two categories: healthy and tumor.
""")