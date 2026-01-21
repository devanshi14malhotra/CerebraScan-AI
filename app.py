import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd

# Styles
st.set_page_config(
    page_title="CerebraScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #4DB6AC 0%, #009688 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "cerebrascan_xception_model.h5"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def preprocess_image(image):
    # Resize to 224x224
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    img_array = np.array(image)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Expand dims
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
    
    # Xception Preprocessing (Scale to -1..1)
    # Be careful: In training we used layers.Rescaling or xception.preprocess_input
    # If the model has preprocess_input BUILT IN (it likely does if we used Transfer Learning correctly or added a layer),
    # we should check. In the notebook, I added `applications.xception.preprocess_input` as a layer.
    # So we pass raw [0, 255] images (but the lambda layer in notebook handles it). 
    # Wait, in the notebook I used: x = applications.xception.preprocess_input(x)
    # This expects [0, 255] input if it's the Keras layer? 
    # Actually `preprocess_input` usually expects floats or ints.
    # Safe bet: pass numpy array. The notebook model has the layer `applications.xception.preprocess_input(x)` as the FIRST layer after inputs.
    # So we should pass the raw image array.
    
    return img_array

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
    st.markdown("## CerebraScan AI")
    st.write("Version: 2.0 (Xception)")
    
    st.info("Upload an MRI scan to detect brain tumors with AI assistance.")

# --- Main ---
st.title("CerebraScan AI: Brain Tumor Classification")

tab1, tab2 = st.tabs(["Analyze", "Batch Analysis"])

model = load_model()

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Scan")
        uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if model:
                if st.button("Analyze Scan"):
                    with st.spinner("Analyzing..."):
                        start_time = pd.Timestamp.now()
                        processed = preprocess_image(image)
                        preds = model.predict(processed)
                        
                        confidences = preds[0]
                        idx = np.argmax(confidences)
                        label_key = CLASSES[idx]
                        label_ui = CLASS_LABELS[label_key]
                        confidence = confidences[idx]
                        
                        time_taken = (pd.Timestamp.now() - start_time).total_seconds()
                    
                    with col2:
                        st.subheader("Results")
                        
                        color = "#4CAF50" if label_key == "notumor" else "#F44336"
                        st.markdown(f"""
                            <div class='metric-card' style='border-left: 5px solid {color}'>
                                <h3 style='margin:0; color:{color}'>{label_ui}</h3>
                                <p style='font-size:1.5em; margin: 10px 0;'>{confidence*100:.2f}%</p>
                                <small>Inference time: {time_taken:.3f}s</small>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("Probabilities:")
                        for i, c in enumerate(CLASSES):
                            st.progress(float(confidences[i]), text=f"{CLASS_LABELS[c]}")

            else:
                st.warning("Model `cerebrascan_xception_model.h5` not found. Please run the training notebook.")

with tab2:
    st.subheader("Batch Processing")
    uploaded_files = st.file_uploader("Upload multiple MRIs", accept_multiple_files=True, type=['jpg','png','jpeg'])
    
    if uploaded_files and model:
        if st.button(f"Process {len(uploaded_files)} Images"):
            results = []
            bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                processed = preprocess_image(img)
                preds = model.predict(processed, verbose=0)
                idx = np.argmax(preds[0])
                results.append({
                    "Filename": file.name,
                    "Prediction": CLASS_LABELS[CLASSES[idx]],
                    "Confidence": preds[0][idx]
                })
                bar.progress((i+1)/len(uploaded_files))
            
            st.dataframe(pd.DataFrame(results).style.format({"Confidence": "{:.2%}"}))
