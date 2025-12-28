import streamlit as st
from PIL import Image
from src.search import HorseRecognizer
import json
import os

# --------------------
# Konfiguration
# --------------------
STALL_MODE = True
CONF_THRESHOLD = 0.8
DELTA_THRESHOLD = 0.1

# --------------------
# Daten laden (GANZ AM ANFANG!)
# --------------------
with open("data/feed_plans.json", encoding="utf-8") as f:
    feed_plans = json.load(f)

# --------------------
# UI
# --------------------
st.set_page_config(page_title="🐴 Stall-Modus", layout="centered")
st.title("🐴 Pferde-Futter-Erkennung")

uploaded_file = st.file_uploader(
    "📸 Foto vom Pferd aufnehmen oder hochladen",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width='stretch')

    recognizer = HorseRecognizer()
    results = recognizer.recognize(image, top_k=3)

    top1, top2 = results[0], results[1]

    unsicher = (
            top1["confidence"] < CONF_THRESHOLD or
            (top1["confidence"] - top2["confidence"]) < DELTA_THRESHOLD
    )

    # --------------------
    # SICHER → automatisch
    # --------------------
    if not unsicher:
        horse = top1["horse"]
        plan = feed_plans[horse]

        st.success(f"🐴 {horse.upper()} erkannt")
        st.markdown(f"""
        ## 🍽️ Futter
        **{plan['futter']}**  
        **Menge:** {plan['menge']}  
        **Zeiten:** {plan['zeiten']}
        """)

    # --------------------
    # UNSICHER → Stall-Modus
    # --------------------
    else:
        st.warning("⚠️ Unsicher – bitte Pferd auswählen")
        st.markdown("## 🐎 Welches Pferd ist es?")

        horses = list(feed_plans.keys())
        cols = st.columns(2)

        for i, horse in enumerate(horses):
            with cols[i % 2]:
                if st.button(
                    f"🐴 {horse.upper()}",
                    use_container_width=True
                ):
                    plan = feed_plans[horse]
                    st.success(f"🍽️ Futter für {horse.upper()}")
                    st.markdown(f"""
                    **{plan['futter']}**  
                    **Menge:** {plan['menge']}  
                    **Zeiten:** {plan['zeiten']}
                    """)

    # --------------------
    # Debug-Ansicht (optional, unten)
    # --------------------
    with st.expander("🧪 Debug: Top-3 Ergebnisse"):
        for i, r in enumerate(results):
            st.write(
                f"{i+1}. **{r['horse']}** – Confidence: `{r['confidence']:.2f}`"
            )

    # --------------------
    # Reset
    # --------------------
    st.divider()
    if st.button("🔄 Neues Pferd", use_container_width=True):
        st.rerun()

