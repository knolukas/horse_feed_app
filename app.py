import streamlit as st
import json
from PIL import Image
from src.search import HorseRecognizer

st.set_page_config(page_title="Pferde Futter App 🐴")

st.title("🐴 Pferde-Futter-Erkennung")

uploaded_file = st.file_uploader(
    "Foto vom Pferd aufnehmen oder hochladen",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Foto", use_container_width=True)

    recognizer = HorseRecognizer()
    horse, confidence = recognizer.recognize(image)

    with open("data/feed_plans.json") as f:
        feed_plans = json.load(f)

    st.subheader(f"Erkanntes Pferd: **{horse}**")
    st.write(f"Confidence: `{confidence:.2f}`")

    if horse in feed_plans:
        plan = feed_plans[horse]
        st.success("🍽️ Futterplan")
        st.write(f"**Futter:** {plan['futter']}")
        st.write(f"**Menge:** {plan['menge']}")
        st.write(f"**Zeiten:** {plan['zeiten']}")
    else:
        st.warning("Kein Futterplan gefunden")
