import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# 🎯 Load Model
# ----------------------------
model = load_model('recyclable_vs_nonrecyclable_cnn.h5')

# ----------------------------
# 🌈 Custom CSS for Professional Look
# ----------------------------
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto:wght@400;500&display=swap');

/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #A8E6CF, #56CCF2);
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
    color: #063970;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #56CCF2, #A8E6CF);
    color: #023047;
    font-family: 'Roboto', sans-serif;
}

/* Header Style */
h1 {
    text-align: center;
    color: #024731;
    font-size: 46px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* Subtitle */
h3 {
    text-align: center;
    color: #1B9C85;
    font-weight: 500;
    font-size: 22px;
    margin-bottom: 25px;
}

/* Upload Box */
.stFileUploader {
    border: 3px dashed #1B9C85 !important;
    border-radius: 10px !important;
    background-color: #E8F9FD !important;
    padding: 15px;
    font-family: 'Roboto', sans-serif;
}

/* Buttons */
.stButton button {
    background-color: #219653;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-family: 'Poppins', sans-serif;
    transition: all 0.3s ease;
}
.stButton button:hover {
    background-color: #1E8449;
    transform: scale(1.05);
}

/* Prediction Boxes */
.result-success {
    background-color: #A8E6CF;
    border-left: 8px solid #219653;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    color: #064420;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
}

.result-error {
    background-color: #FFD6A5;
    border-left: 8px solid #E74C3C;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    color: #7B241C;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    color: #0B5345;
    font-size: 14px;
    font-family: 'Roboto', sans-serif;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# 🌍 Title Section
# ----------------------------
st.markdown("<h1>♻️ Recyclable vs Non-Recyclable Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-powered waste identification for a sustainable future 🌱</h3>", unsafe_allow_html=True)
st.write("Upload an image below to find out whether it is **Recyclable** or **Non-Recyclable**.")

# ----------------------------
# 📁 Upload Image
# ----------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ----------------------------
# 🔍 Prediction Logic
# ----------------------------
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Handle unreadable formats (e.g., WebP)
        if img is None:
            uploaded_file.seek(0)
            pil_img = Image.open(uploaded_file).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="📸 Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = cv2.resize(img, (128, 128))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # Predict
        prediction = model.predict(img_resized)[0][0]
        confidence = round(prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100, 2)

        st.write("### 🧠 Prediction Result:")
        if prediction >= 0.5:
            st.markdown(f"<div class='result-error'>🚯 Non-Recyclable Waste<br><b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-success'>✅ Recyclable Waste ♻️<br><b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error reading image: {e}")

# ----------------------------
# 🧭 Sidebar
# ----------------------------
with st.sidebar:
    st.header("🔖 Project Info")
    st.write("""
    **Project Title:** Recyclable vs Non-Recyclable Classifier  
    **Model:** Convolutional Neural Network (CNN)  
    **Framework:** TensorFlow 2.13  
    **Developer:** Manohar Madhu  
    **Goal:** Promote sustainable waste management 🌍
    """)

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("<div class='footer'>Made with 💚 for sustainability | © 2025 Manohar Madhu</div>", unsafe_allow_html=True)
