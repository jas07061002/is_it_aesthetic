import streamlit as st
import requests
from PIL import Image
import io
import os
import time

# ✅ Function to get API Key (Handles missing/empty cases)
def get_api_key():
    HF_API_KEY = os.getenv("HF_API_KEY") or st.secrets.get("HF_API_KEY", "").strip()
    
    if not HF_API_KEY:
        st.error("❌ Hugging Face API Key is missing! Add it to `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets.")
        st.stop()
    
    return HF_API_KEY

# ✅ Load API key securely
HF_API_KEY = get_api_key()
API_URL = "https://api-inference.huggingface.co/models/cafeai/cafe_aesthetic"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Function to classify images with error handling
def classify_image(image):
    # ✅ Convert to JPEG (reduces size)
    image = image.convert("RGB")  # Ensure correct format
    image = image.resize((256, 256))  # Resize to 256x256

    # ✅ Convert to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG", quality=85)  # Compress

    # ✅ Send API request
    response = requests.post(API_URL, headers=headers, data=image_bytes.getvalue())

    # ✅ Handle API response
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 503:
        # Model is still loading, wait & retry
        try:
            error_data = response.json()
            estimated_time = error_data.get("estimated_time", 30)
            st.warning(f"Model is still loading. Waiting for {estimated_time:.2f} seconds...")
            time.sleep(int(estimated_time))
            return classify_image(image)  # Retry request
        except requests.exceptions.JSONDecodeError:
            return {"error": "API response was not in JSON format"}
    else:
        return {
            "error": f"API request failed with status code {response.status_code}",
            "details": response.text
        }

# ✅ Streamlit UI
st.title("Anime vs. Real Aesthetic Classifier 🌟")
st.write("Upload an image to check if it's **Anime or Real** using the `cafeai/cafe_aesthetic` model!")

uploaded_file = st.file_uploader("Choose an image (Max 3MB)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # ✅ File size check
    if uploaded_file.size > 4 * 1024 * 1024:  # 3MB limit
        st.error("❌ File too large! Please upload an image smaller than 3MB.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        result = classify_image(image)

        if "error" in result:
            st.error(f"{result['error']}\n\nDetails: {result.get('details', 'No additional info')}")
        else:
            st.subheader(f"Prediction: {result[0]['label']} 🎉")
            st.write(f"Confidence: {result[0]['score']:.2f}")
