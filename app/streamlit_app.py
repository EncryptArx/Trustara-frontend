import streamlit as st
import httpx
from pathlib import Path


st.set_page_config(page_title="DeepSecure MVP", layout="centered")
st.title("DeepSecure MVP")
st.caption("Upload an image, video, or audio file to detect deepfakes")


try:
    backend_url = st.secrets["backend_url"]  # requires secrets.toml; use default if missing
except Exception:
    backend_url = "http://127.0.0.1:8000"

uploaded = st.file_uploader("Upload media", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "wav", "mp3", "m4a"]) 
analyze = st.button("Analyze")

if analyze:
    if not uploaded:
        st.warning("Please upload a media file first.")
    else:
        with st.spinner("Analyzing..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
            try:
                resp = httpx.post(f"{backend_url}/analyze", files=files, timeout=120)
                if resp.status_code != 200:
                    st.error(f"Analysis failed: {resp.text}")
                else:
                    data = resp.json()
                    st.subheader("Result")
                    st.write(f"Label: {data.get('result', 'unknown').upper()}")
                    st.write(f"Confidence: {data.get('confidence', 0.0)}%")
                    st.write(f"Media type: {data.get('media_type')}")
                    st.write(f"Timestamp: {data.get('timestamp')}")
                    st.write(f"Model: {data.get('model_version')}")

                    geo = data.get("geo_tag") or {}
                    if geo:
                        st.write(f"Geo-tag: {geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')}")

                    exp_path = data.get("explanation_path")
                    if exp_path and Path(exp_path).exists():
                        suffix = Path(exp_path).suffix.lower()
                        if suffix in [".png", ".jpg", ".jpeg"]:
                            st.image(exp_path, caption="Grad-CAM explanation")
                        else:
                            st.write(f"Explanation saved at: {exp_path}")
            except Exception as e:
                st.exception(e)


