import streamlit as st
from PIL import Image
from src.search import HorseRecognizer
import json
import os
from datetime import datetime
from PIL import Image
from src.build_index import build_index

# --------------------
# Konfiguration
# --------------------
STALL_MODE = True
CONF_THRESHOLD = 0.8
DELTA_THRESHOLD = 0.1
DATA_DIR = "data/images"

def save_images(images, horse_name):
    horse_name = horse_name.lower().strip()

    save_dir = os.path.join(DATA_DIR, horse_name)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0

    for img in images:
        if img is None:
            continue

        image = Image.open(img).convert("RGB")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(save_dir, f"{ts}.jpg")
        image.save(path)
        saved += 1

    return saved


# --------------------
# Daten laden (GANZ AM ANFANG!)
# --------------------
with open("data/feed_plans.json", encoding="utf-8") as f:
    feed_plans = json.load(f)

# --------------------
# UI
# --------------------
st.set_page_config(page_title="🐴 Pferde-App", layout="centered")

st.divider()

#******************************
#******************************
st.subheader("➕ Pferd anlernen / Fotos hinzufügen")

camera_photo = st.file_uploader(
    "📸 Foto vom Pferd aufnehmen oder hochladen",
    type=["jpg", "png", "jpeg"],
    key="camera_training_upload")

horse_name = st.text_input(
    "🐴 Pferdename eingeben (neu oder bestehend)",
    placeholder="z. B. Megapferd"
)

if camera_photo and horse_name:
    horse_name = horse_name.strip().lower()

    save_dir = f"data/images/{horse_name}"
    os.makedirs(save_dir, exist_ok=True)

    from datetime import datetime
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + horse_name +".jpg"
    save_path = os.path.join(save_dir, filename)

    image = Image.open(camera_photo).convert("RGB")
    image.save(save_path)

    st.success(f"✅ Foto für **{horse_name.upper()}** gespeichert")
    st.caption("📂 " + save_path)

    st.info("ℹ️ Index wird beim nächsten Neuaufbau aktualisiert")


#******************************
#******************************
st.subheader("🐴 Pferd anlernen")

horse_name = st.text_input(
    "Name des Pferdes",
    placeholder="z. B. Jolly",
    key="horse_name_input"
)

if not horse_name:
    st.info("Bitte zuerst einen Pferdenamen eingeben.")
    st.stop()

st.markdown("### 📁 📸 Fotos aufnehmen/hochladen (mehrere möglich)")

uploaded_files = st.file_uploader(
    "Mehrere Fotos auswählen",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="multi_upload"
)

images_to_save = []

if uploaded_files:
    images_to_save.extend(uploaded_files)

if images_to_save:
    if st.button("💾 Fotos speichern", use_container_width=True):
        save_images(images_to_save, horse_name)
        st.success(f"✅ {len(images_to_save)} Fotos für {horse_name} gespeichert")


if st.button("🔁 Index neu bauen", use_container_width=True):
    with st.spinner("Index wird neu gebaut..."):
        build_index()
    st.success("✅ Index erfolgreich neu gebaut")
#******************************
#******************************
st.subheader("🐴 Pferde-Futter-Erkennung")

uploaded_file = st.file_uploader(
    "📸 Foto vom Pferd aufnehmen oder hochladen",
    type=["jpg", "png", "jpeg"],
    key="app_use_upload"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width='stretch')

    recognizer = HorseRecognizer()
    results = recognizer.recognize(image, top_k=3)

    st.write("DEBUG – Results:", results)

    if len(results) == 0:
        st.error("❌ Kein bekanntes Pferd erkannt")
        st.stop()

    top1 = results[0]
    top2 = results[1] if len(results) > 1 else None

    unsicher = (
            top1["confidence"] < CONF_THRESHOLD or
            (top2 is not None and
             (top1["confidence"] - top2["confidence"]) < DELTA_THRESHOLD)
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

