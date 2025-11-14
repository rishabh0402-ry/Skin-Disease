# app.py
import os
import tempfile
from io import BytesIO

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import timm

# NOTE: ultralytics might log a lot; it's OK
from ultralytics import YOLO

# ---------- CONFIG ----------
YOLO_WEIGHTS_PATH = "models/best.pt"                 # <<-- change if needed
EFF_WEIGHTS_PATH = "models/efficientnet_best.pth"    # <<-- change if needed

NUM_CLASSES = 7

# Friendly class names: keep consistent with your training labels
CLASS_NAMES = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]

# ---------- Caching model loads ----------
@st.cache_resource(show_spinner=False)
def load_yolo(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLO weights not found at {path}")
    model = YOLO(path)
    return model

@st.cache_resource(show_spinner=False)
def load_efficientnet(path: str, device: str = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"EfficientNet weights not found at {path}")
    # create model architecture
    eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    # load weights
    state = torch.load(path, map_location=device, weights_only=False)
    # If the saved file is a state_dict or a raw model path, handle both
    if isinstance(state, dict) and ("state_dict" in state):
        state = state["state_dict"]
    eff.load_state_dict(state)
    eff.to(device)
    eff.eval()
    return eff

# ---------- Preprocessing ----------
eff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- Utility functions ----------
def pil_to_cv2(pil_image: Image.Image):
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    # convert RGB to BGR for cv2
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray):
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_box_and_label(img_cv2: np.ndarray, box, label_text: str):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # rectangle
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # text background
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img_cv2, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img_cv2, label_text, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img_cv2

# ---------- Prediction pipeline ----------
def predict_on_image_file(file_path: str, yolo_model, eff_model, device="cpu"):
    """
    file_path: path to image file (jpg/png)
    returns: dict with keys: original_pil, annotated_pil, prediction_text, class_idx, confidence
    """
    # Run YOLO detection
    try:
        results = yolo_model(file_path)[0]       # ultralytics returns list-like results
    except Exception as e:
        return {"error": f"YOLO inference failed: {e}"}

    # If no boxes detected
    if len(results.boxes) == 0:
        return {"error": "No lesion detected by YOLO."}

    # Use the first bounding box (you can modify to pick largest/confident detection)
    box_xyxy = results.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    # read image
    img_cv2 = cv2.imread(file_path)
    if img_cv2 is None:
        return {"error": "Failed to read uploaded image."}

    h, w = img_cv2.shape[:2]
    # Clip coordinates to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # if bounding box invalid
    if x2 <= x1 or y2 <= y1:
        return {"error": "Invalid bounding box from YOLO."}

    crop = img_cv2[y1:y2, x1:x2]
    if crop.size == 0:
        return {"error": "Cropped region is empty."}

    # Convert crop to PIL -> transform -> tensor -> batch
    crop_pil = cv2_to_pil(crop)
    tensor = eff_transform(crop_pil).unsqueeze(0).to(device)

    # EfficientNet prediction
    with torch.no_grad():
        outputs = eff_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        conf_val = float(conf.cpu().numpy()[0])
        pred_idx = int(pred.cpu().numpy()[0])

    label_text = f"{CLASS_NAMES[pred_idx]} ({conf_val*100:.2f}%)"

    # Annotate original image
    annotated = img_cv2.copy()
    draw_box_and_label(annotated, (x1, y1, x2, y2), label_text)

    res = {
        "original_pil": cv2_to_pil(img_cv2),
        "annotated_pil": cv2_to_pil(annotated),
        "prediction_text": label_text,
        "class_idx": pred_idx,
        "confidence": conf_val
    }
    return res

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Skin Lesion Detector", layout="centered")

st.title("Skin Lesion Detection & Classification")
st.write("Upload an image of a skin lesion. The app will localize the lesion (YOLO) and classify it (EfficientNet).")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("Model files (placed in `models/`) :")
    st.text(f"YOLO: {YOLO_WEIGHTS_PATH}")
    st.text(f"EfficientNet: {EFF_WEIGHTS_PATH}")
    st.markdown("If your files have different names, edit the top of `app.py` accordingly.")

show_models = st.checkbox("Show model load status / (reload models)", value=False)

# Load models (once cached)
device = "cpu"
try:
    with st.spinner("Loading YOLO model..."):
        yolo_model = load_yolo(YOLO_WEIGHTS_PATH)
    with st.spinner("Loading EfficientNet model..."):
        eff_model = load_efficientnet(EFF_WEIGHTS_PATH, device=device)
    if show_models:
        st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.getbuffer())
    tfile.flush()
    tfile.close()
    img_path = tfile.name

    st.image(Image.open(img_path), caption="Uploaded image", use_column_width=True)

    if st.button("Run prediction"):
        with st.spinner("Running detection and classification..."):
            result = predict_on_image_file(img_path, yolo_model, eff_model, device=device)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(result["prediction_text"])
            st.write(f"Predicted class index: {result['class_idx']}, confidence: {result['confidence']:.4f}")

            st.markdown("**Annotated output**")
            st.image(result["annotated_pil"], use_column_width=True)

            # allow download of annotated image
            buf = BytesIO()
            result["annotated_pil"].save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button("Download annotated image", data=byte_im, file_name="annotated.jpg", mime="image/jpeg")

# Footer / tips
st.markdown("---")
st.markdown("**Tips:**")
st.markdown("- Place model files in a `models/` folder next to this `app.py` (or update the paths above).")
st.markdown("- If your EfficientNet was trained with a different architecture/name, change `timm.create_model(...)` accordingly.")
st.markdown("- On Streamlit Cloud make sure `requirements.txt` contains `ultralytics`, `timm`, `torch`, `opencv-python-headless`, `pillow`, `streamlit` etc.")
