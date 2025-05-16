import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import pickle
import pandas as pd
import io

# Load pre-trained models
@st.cache_resource
def load_models():
    # Load VGG11 model
    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    vgg11.classifier = nn.Sequential(*list(vgg11.classifier.children())[:-1])  # Remove last layer
    
    # Load PCA and MLP models
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('best_mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)
    
    return vgg11, pca, mlp

vgg11, pca_model, mlp_model = load_models()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = vgg11(img_tensor)
    return features.numpy().flatten()

# Function to reduce dimensions
def reduce_dimensions(features):
    return pca_model.transform(features.reshape(1, -1))[0]

# Function to predict class
def predict_class(reduced_features):
    return mlp_model.predict(reduced_features.reshape(1, -1))[0]

# Streamlit app layout
st.title("Burn Injury Classification System")

# Initialize session state
if 'features' not in st.session_state:
    st.session_state.features = None
if 'reduced_features' not in st.session_state:
    st.session_state.reduced_features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Upload image
uploaded_file = st.file_uploader("Upload a burn injury image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process buttons in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Extract Features (4096D)"):
            st.session_state.features = extract_features(image)
            st.success("Features extracted successfully!")
    
    with col2:
        if st.button("Reduce to 20 Dimensions"):
            if st.session_state.features is not None:
                st.session_state.reduced_features = reduce_dimensions(st.session_state.features)
                st.success("Dimensionality reduced successfully!")
            else:
                st.warning("Please extract features first!")
    
    with col3:
        if st.button("Predict Class"):
            if st.session_state.reduced_features is not None:
                st.session_state.prediction = predict_class(st.session_state.reduced_features)
                st.success(f"Predicted class: {st.session_state.prediction}")
            else:
                st.warning("Please reduce dimensions first!")

# Results display section
st.header("Results")

# Feature extraction results
st.subheader("Feature Extraction Results")
if st.session_state.features is not None:
    st.write(f"Feature vector shape: {st.session_state.features.shape}")
    
    # Download features
    df_features = pd.DataFrame(st.session_state.features.reshape(1, -1))
    csv = df_features.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Features as CSV",
        data=csv,
        file_name="extracted_features.csv",
        mime="text/csv"
    )
else:
    st.info("No features extracted yet.")

# Dimensionality reduction results
st.subheader("Dimensionality Reduction Results")
if st.session_state.reduced_features is not None:
    st.write(f"Reduced feature vector shape: {st.session_state.reduced_features.shape}")
    st.write("Reduced features:")
    st.write(st.session_state.reduced_features)
else:
    st.info("No dimensionality reduction performed yet.")

# Prediction results
st.subheader("Prediction Results")
if st.session_state.prediction is not None:
    st.write(f"Predicted burn class: {st.session_state.prediction}")
    # Add class descriptions if available
    class_descriptions = {
        0: "未检测出烧烫伤",
        1: "浅二度烫伤",
        2: "深二度烫伤",
        3: "三度烫伤",
        4: "电击烧伤",
        5: "火焰烧伤"
    }
    st.write(f"Description: {class_descriptions.get(st.session_state.prediction, 'Unknown class')}")
else:
    st.info("No prediction made yet.")

# Instructions section
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a burn injury image (JPG/PNG)
2. Click 'Extract Features' to get 4096D features
3. Click 'Reduce to 20 Dimensions' for PCA
4. Click 'Predict Class' for final classification
""")

# Model information
st.sidebar.header("Model Information")
st.sidebar.write("""
- **Feature Extraction**: VGG11 (pretrained on ImageNet)
- **Dimensionality Reduction**: PCA (20 components)
- **Classifier**: MLP (pre-trained)
""")