# 🧠 CerebraScan AI

### Brain Tumor MRI Classification System

![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)

**CerebraScan AI** is a deep learning-based medical imaging application designed to classify brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. 

Built with **TensorFlow/Keras** (using Transfer Learning with **Xception**) and deployed via **Streamlit**, this tool provides a fast, accurate, and user-friendly interface for medical professionals and researchers.

---

## 🚀 Features

- **High Accuracy**: Utilizes the Xception architecture (pre-trained on ImageNet) to achieve **>92% accuracy**.
- **Real-Time Analysis**: Instant classification of uploaded MRI images.
- **Batch Processing**: Analyze multiple scans simultaneously with summary statistics.
- **Interactive UI**: Premium, dark-themed dashboard built with Streamlit.
- **Privacy Focused**: Processes images locally (when running locally); no data storage on external servers.

## 📂 Project Structure

```

app.py # Streamlit Application
CerebraScan_Workflow.ipynb # End-to-End Jupyter Notebook
train_temp.py # Quick training script
cerebrascan_xception_model.h5 # Trained Model Artifact
brain_mri_dataset # Dataset (Train/Test splits)
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

### 1. Run the Web Application
To start the CerebraScan interface:
```bash
cd notebook_version
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 2. Train/Retrain the Model
You can use the Jupyter Notebook for a step-by-step walkthrough or the python script:

**Option A: Jupyter Notebook**
Open `notebook_version/CerebraScan_Workflow.ipynb` in Jupyter Lab or VS Code to explore the data ecosystem, training process, and evaluation metrics.

**Option B: Python Script**
```bash
cd notebook_version
python train_temp.py
```

## 🧠 Model Details

-   **Architecture**: Xception (Transfer Learning)
-   **Input Shape**: 224x224x3
-   **Preprocessing**: Rescaling [-1, 1]
-   **Classes**:
    -   Glioma
    -   Meningioma
    -   Pituitary
    -   No Tumor

## 📈 Performance

The model typically achieves:
-   **Validation Accuracy**: ~90-95%
-   **Test Accuracy**: >92%
-   **Inference Time**: <200ms per image (on GPU)

#### ⚠️ Disclaimer
**CerebraScan AI is for educational and research purposes only.** It is **not** a certified medical device and should not be used as a primary diagnostic tool. Always consult a qualified radiologist or medical professional for diagnosis.




