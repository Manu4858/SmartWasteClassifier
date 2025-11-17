# ♻️ Recyclable vs Non-Recyclable Waste Classification (CNN + Streamlit)

### 🌱 Deep Learning project for sustainability and waste segregation

---

## 🌍 Overview

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify images of waste materials as either:
- ✅ **Recyclable**
- 🚯 **Non-Recyclable**

It also features an interactive **Streamlit web interface** with a professional **blue–green sustainability theme**, allowing users to upload an image and instantly see the classification result.

---

## 🧠 Key Features

- 🧩 Deep Learning model trained with **TensorFlow 2.13**
- 📸 Real-time prediction using **OpenCV**
- 💻 Interactive and stylish **Streamlit UI**
- 🌈 Blue–green sustainability-themed design
- ⚙️ Automatic dataset splitting (train / val / test)
- 🔍 Displays prediction confidence percentage
- ⚡ Works on **Mac, Windows, or Google Colab**

---

## 📁 Folder Structure

recyclable_classification_cnn_project/
│
├── data/
│ ├── train/
│ │ ├── recyclable/
│ │ └── non_recyclable/
│ ├── val/
│ │ ├── recyclable/
│ │ └── non_recyclable/
│ └── test/
│ ├── recyclable/
│ └── non_recyclable/
│
├── app.py # Streamlit Web App (main interface)
├── train.py # Model training script
├── predict.py # Command-line prediction script
├── split_data.py # Script to auto-split dataset
├── recyclable_vs_nonrecyclable_cnn.h5 # Trained CNN model
├── requirements.txt # Dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🧩 Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/recyclable-vs-nonrecyclable-classification.git
cd recyclable-vs-nonrecyclable-classification
2️⃣ Create a Python 3.10 virtual environment
bash
Copy code
python3.10 -m venv env310
source env310/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🧪 Model Training
Step 1: Organize your dataset
Place images under:

bash
Copy code
data/train/recyclable/
data/train/non_recyclable/
Step 2: Split into train/val/test automatically
bash
Copy code
python split_data.py
Step 3: Train your CNN model
bash
Copy code
python train.py
The model will be saved as:

Copy code
recyclable_vs_nonrecyclable_cnn.h5
🔍 Prediction (Command Line)
To test a single image directly:

bash
Copy code
python predict.py data/test/recyclable/R_10000.jpg
Output example:

makefile
Copy code
✅ Recyclable
Confidence: 94.6%
💻 Run the Streamlit Web App
Once the model is trained, launch the web interface:

bash
Copy code
streamlit run app.py
Then open your browser at:
👉 http://localhost:8501