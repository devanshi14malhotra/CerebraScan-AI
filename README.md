# 🧠 CerebraScan AI

### Brain Tumor MRI Classification System

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://cerebra-scan-ai.streamlit.app/)
![Status](https://img.shields.io/badge/Status-Live-success)

**CerebraScan AI** is a deep learning-based medical imaging application designed to classify brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. 

---

## 🚀 Features

- **High Accuracy**: Utilizes the Xception architecture (pre-trained on ImageNet) to achieve **>90% accuracy**.
- **Real-Time Analysis**: Instant classification of uploaded MRI images.
- **Batch Processing**: Analyze multiple scans simultaneously with summary statistics.
- **Interactive UI**: Premium, dark-themed dashboard built with Streamlit.
- **Privacy Focused**: Processes images locally (when running locally); no data storage on external servers.

## 📂 Project Structure

```
app.py # Streamlit Application
CerebraScan_AI.ipynb # End-to-End Jupyter Notebook
train_temp.py # Quick training script
*.h5 # Trained Model Artifacts
requirements.txt # Dependencies
README.md # Documentation
```

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/devanshi14malhotra/cerebrascan-ai.git
    cd cerebrascan-ai
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### 1. Try out the application using the deployed link
```
https://cerebra-scan-ai.streamlit.app/
```

### 2. Run the Web Application
To start the CerebraScan interface:
```bash
streamlit run app.py
```

### 3. Train/Retrain the Model
You can use the Jupyter Notebook for a step-by-step walkthrough or the Python script:

**Option A: Jupyter Notebook**
Open `CerebraScan_AI.ipynb` in Jupyter Lab or VS Code or Google Colab to explore the data ecosystem, training process, and evaluation metrics.

**Option B: Python Script**
```bash
python train_temp.py
```

## 📈 Performance

The model typically achieves:
-   **Validation Accuracy**: ~91-95%
-   **Test Accuracy**: ~90%
-   **Inference Time**: Faster on GPU

#### ⚠️ Disclaimer
**CerebraScan AI is for educational and research purposes only.** It is **not** a certified medical device and should not be used as a primary diagnostic tool. Always consult a qualified radiologist or medical professional for diagnosis.




