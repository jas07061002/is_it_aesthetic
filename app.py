import streamlit as st
import requests
from PIL import Image
import io
import os
import time
import logging

# Set up logging
logging.basicConfig(
    filename="app.log",  # Log file
    level=logging.DEBUG,  # Log level (INFO, DEBUG, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
)

#  Load API key securely
#HF_API_KEY = hf_api_key


# Log API key retrieval
#HF_API_KEY = st.secrets.get("HF_API_KEY", "").strip()
HF_API_KEY = st.secrets.get("secrets", {}).get("HF_API_KEY")

if not HF_API_KEY:
    logging.error("Hugging Face API Key is missing! Check secrets.toml or Streamlit Cloud Secrets.")
    st.error("Hugging Face API Key is missing! Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()
else:
    logging.info("Hugging Face API Key loaded successfully.")

API_URL = "https://api-inference.huggingface.co/models/cafeai/cafe_aesthetic"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to classify images with error handling
def classify_image(image):
    logging.info("Starting image classification...")

    try:
        # Convert to JPEG (reduces size)
        image = image.convert("RGB")  # Ensure correct format
        image = image.resize((128, 128))  # Resize to 128x128

        # Convert to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG", quality=85)  # Compress

        # Send API request
        logging.info("Sending image to Hugging Face API...")
        response = requests.post(API_URL, headers=headers, data=image_bytes.getvalue())

        # Handle API response
        if response.status_code == 200:
            logging.info("API request successful!")
            return response.json()
        elif response.status_code == 503:
            # Model is still loading, wait & retry
            logging.warning("Model is still loading. Retrying after estimated wait time.")
            try:
                error_data = response.json()
                estimated_time = error_data.get("estimated_time", 30)
                time.sleep(int(estimated_time))
                return classify_image(image)  # Retry request
            except requests.exceptions.JSONDecodeError:
                logging.error("API response was not in JSON format.")
                return {"error": "API response was not in JSON format"}
        else:
            logging.error(f"API request failed! Status Code: {response.status_code}, Response: {response.text}")
            return {
                "error": f"API request failed with status code {response.status_code}",
                "details": response.text
            }

    except Exception as e:
        logging.exception(f"Unexpected error during classification: {e}")
        return {"error": "An unexpected error occurred", "details": str(e)}

# Streamlit UI
st.title("Anime vs. Real Aesthetic Classifier 🌟")
st.write("Upload an image to check if it's **Anime or Real**!")

uploaded_file = st.file_uploader("Choose an image (Max 3MB)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # File size check
        if uploaded_file.size > 4 * 1024 * 1024:  # 3MB limit
            logging.warning("Uploaded file is too large.")
            st.error("File too large! Please upload an image smaller than 3MB.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            logging.info("Image uploaded successfully. Processing...")
            result = classify_image(image)

            if "error" in result:
                logging.error(f"Classification error: {result['error']}")
                st.error(f"{result['error']}\n\nDetails: {result.get('details', 'No additional info')}")
            else:
                st.subheader(f"Prediction: {result[0]['label']} 🎉")
                st.write(f"Confidence: {result[0]['score']:.2f}")

    except Exception as e:
        logging.exception(f"Unexpected error in Streamlit UI: {e}")
        st.error("An unexpected error occurred. Check the logs for details.")

# Display logs in Streamlit for debugging
#if st.button("Show Logs"):
#    with open("app.log", "r") as log_file:
#        st.text(log_file.read())

