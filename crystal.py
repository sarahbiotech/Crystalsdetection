import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np


st.set_page_config(page_title="Crystal AI Detection", layout="wide")


@st.cache_resource
def load_model():
    model = YOLO("crystal.pt")
    return model

model = load_model()

col1, col2 = st.columns([1, 10])

with col1:
    st.image("urine.png", width=2000)

with col2:
    st.title("Crystal AI Detection")
st.subheader("YOLOv8-based Urine Crystal Detection Model")

st.markdown("""
Upload a microscopic image to detect urine crystals automatically using a trained YOLOv8 model.
""")

st.divider()


with st.sidebar:
    st.header("📌 Model Info")

    with st.expander("Model Details"):
        st.write("""
- Framework: YOLOv8 (Ultralytics)
- Task: Object Detection
- Type: Urine Crystal Detection
        """)

    with st.expander("Supported Crystals"):
        st.write("""
- Calcium Oxalate Dihydrate  
- Calcium Oxalate Monohydrate (Ovoid)  
- Phosphate
        """)

    with st.expander("Performance Metrics"):
        st.write("""
- Precision: 0.718  
- Recall: 0.665  
- mAP50: 0.706  
- mAP50-95: 0.489  
        """)


st.subheader("⚙️ Detection Settings")

conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

st.divider()


uploaded_file = st.file_uploader("📤 Upload Microscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

   
    with col1:
        st.markdown("##  Before Prediction")
        st.image(image, use_container_width=True)

    
    with col2:
        st.markdown("##  prediction Preview")
        st.info("Image loaded successfully and ready for analysis")

        run = st.button("🔬 Analysis")


    if run:

        st.markdown("## 📊 Results")

      
        results = model.predict(source=np.array(image), conf=conf)

        result_img = results[0].plot()

        st.image(result_img, caption="Detected Crystals", use_container_width=True)

      
        boxes = results[0].boxes

        counts = {}

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in counts:
                counts[class_name] += 1
            else:
                counts[class_name] = 1

       
        st.subheader("Crystal Detection Summary")

        if counts:
            total = 0

            for name, value in counts.items():
                st.write(f"• {name}: {value}")
                total += value

            st.success(f"Total Crystals Detected: {total}")

        else:
            st.warning("No crystals detected in this image.")


st.divider()
st.markdown(
    "<center><b>Crystal AI Application</b> | Developed by Sarah © 2026</center>",
    unsafe_allow_html=True
)