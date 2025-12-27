import streamlit as st
from PIL import Image
from src.search import HorseRecognizer
import json

st.title("🐴 Pferde-Futter-Erkennung (Debug-Ansicht)")

uploaded_file = st.file_uploader("Foto vom Pferd hochladen", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Foto", use_container_width=True)

    recognizer = HorseRecognizer()
    results = recognizer.recognize(image, top_k=3)

    st.subheader("✅ Top-Ergebnisse")
    for i, r in enumerate(results):
        st.write(f"{i+1}. Pferd: **{r['horse']}** – Confidence: `{r['confidence']:.2f}`")

    # Optional: Futterplan des Top-1
    with open("data/feed_plans.json") as f:
        feed_plans = json.load(f)

    top_horse = results[0]['horse']
    if top_horse in feed_plans:
        plan = feed_plans[top_horse]
        st.success("🍽️ Futterplan")
        st.write(f"**Futter:** {plan['futter']}")
        st.write(f"**Menge:** {plan['menge']}")
        st.write(f"**Zeiten:** {plan['zeiten']}")
    else:
        st.warning("Kein Futterplan gefunden")
