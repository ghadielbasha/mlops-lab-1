import os
import requests
import streamlit as st

INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://127.0.0.1:8000"
)

st.title("Food-11 Classifier")

st.write("Upload a food image to classify it using our ResNet18 model.")

uploaded = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    st.image(uploaded, width=300)

    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type
        )
    }

    try:
        response = requests.post(
            f"{INFERENCE_URL}/predict",
            files=files,
            timeout=60
        )

        if response.ok:
            result = response.json()

            st.success(
                f"Prediction: {result['category']}"
            )

            st.write(
                f"Confidence: {result['confidence']:.1%}"
            )
        else:
            st.error(
                f"Inference error: {response.status_code}"
            )

    except requests.RequestException as error:
        st.error(f"Cannot reach inference service: {error}")
