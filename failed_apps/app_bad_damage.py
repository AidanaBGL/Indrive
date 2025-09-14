import io
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

# =============== PAGE CONFIG ===============
st.set_page_config(page_title="Webcam Car Side Verifier + Face Gate + Damage Check", layout="wide")

# =============== CLASS MAPS / RULES ===============
CLASS_MAP = {
    0: "back_bumper",
    1: "hood",
    2: "back_glass",
    3: "back_left_door",
    4: "trunk",
    5: "back_light",
    6: "back_right_door",
    7: "roof",
    8: "front_bumper",
    9: "grille",
    10: "front_glass",
    11: "front_left_door",
    12: "bonnet",
    13: "front_light",
    14: "front_right_door",
    15: "wheel",
    16: "license_plate",
    17: "left_mirror",
    18: "spoiler",
    19: "right_mirror",
}

SIDE_RULES = {
    "front": {8, 10, 13},   # front_bumper, front_glass, front_light
    "left":  {11, 3, 17},   # front_left_door, back_left_door, left_mirror
    "back":  {0, 2, 5},     # back_bumper, back_glass, back_light
    "right": {6, 14, 19},   # back_right_door, front_right_door, right_mirror
}

SIDE_ORDER = ["front", "left", "back", "right"]
SIDE_LABELS = {"front": "Front", "left": "Left", "back": "Back", "right": "Right"}

def ids_to_names(ids):
    return sorted({CLASS_MAP.get(int(i), f"class_{int(i)}") for i in ids})

# =============== UTIL: PIL <-> CV2 ===============
def pil_to_cv(image: Image.Image):
    arr = np.array(image.convert("RGB"))
    return arr[:, :, ::-1].copy()

# =============== DEEPFACE GATE ===============
from deepface import DeepFace  # pip install deepface retina-face opencv-python

def verify_two_faces_in_one(image: Image.Image) -> Tuple[bool, Optional[float], Optional[str], int]:
    """
    Возвращает (matched, distance, error_message, num_faces)
    matched=True только если в кадре ровно 2 лица (человек + фото на документе) и они совпадают.
    """
    try:
        arr_bgr = pil_to_cv(image)
        faces = DeepFace.extract_faces(
            img_path=arr_bgr,
            detector_backend="retinaface",
            enforce_detection=False
        )
        num_faces = len(faces)

        if num_faces < 2:
            return False, None, (
                "На снимке найдено меньше двух лиц. Держите документ ближе к камере и следите, "
                "чтобы портрет на документе был хорошо виден."
            ), num_faces

        if num_faces > 2:
            return False, None, (
                "На снимке обнаружено больше двух лиц. Уберите лишних людей из кадра."
            ), num_faces

        # Ровно 2 лица → сравниваем
        f1 = faces[0]["face"]
        f2 = faces[1]["face"]
        result = DeepFace.verify(
            img1_path=f1,
            img2_path=f2,
            model_name="Facenet512",
            detector_backend="opencv",   # детектор не нужен (у нас уже кропы), но параметр обязателен
            enforce_detection=False
        )
        matched = bool(result.get("verified", False))
        distance = float(result.get("distance", 0.0))
        return matched, distance, None, num_faces

    except Exception as e:
        return False, None, f"Ошибка при анализе лиц: {e}", 0

# =============== MODEL LOADING (YOLO) ===============
@st.cache_resource(show_spinner=True)
def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        st.error("Ultralytics not installed. Run: `pip install ultralytics`")
        return None
    weights_path = "/hdd/home/ABaglanova/indrive_hack/car_seg/runs/segment/train2/weights/best.pt"
    try:
        model = YOLO(weights_path)
        # Use model's class names if available to avoid mismatches
        try:
            if hasattr(model, "names"):
                names = {int(i): str(n) for i, n in model.names.items()}
                if names:
                    global CLASS_MAP
                    CLASS_MAP = names
        except Exception:
            pass
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO weights at {weights_path}\n\n{e}")
        return None

yolo_model = load_yolo()

# =============== DAMAGE PIPELINE (HF Transformers) ===============
@st.cache_resource(show_spinner=True)
def load_damage_pipe():
    try:
        from transformers import pipeline
    except Exception as e:
        st.error("Transformers is not installed. Run: `pip install transformers pillow`")
        return None
    try:
        return pipeline("image-classification", model="beingamit99/car_damage_detection")
    except Exception as e:
        st.error(f"Failed to load damage model: {e}")
        return None

# =============== SIDEBAR CONTROLS ===============
with st.sidebar:
    st.markdown("### YOLO Inference Settings")
    conf_thres = st.slider("YOLO confidence", 0.05, 0.75, 0.25, 0.01)
    img_size   = st.slider("YOLO image size", 320, 1280, 640, 32)
    st.markdown("---")
    st.markdown("### Damage Detection Settings")
    damage_threshold = st.slider("Damage confidence threshold", 0.10, 0.95, 0.50, 0.01,
                                 help="If the top damage score ≥ this value on any side, the car is NOT acceptable for taxi.")
    st.caption("Model: beingamit99/car_damage_detection")

# =============== SESSION STATE ===============
if "step" not in st.session_state:
    # step 0 = face gate, 1..4 = sides, 5 = summary
    st.session_state.step = 0
if "sides" not in st.session_state:
    st.session_state.sides = {
        k: {"true": None, "detected_ids": set(), "shot": None, "annotated": None}
        for k in SIDE_ORDER
    }
if "face_gate" not in st.session_state:
    st.session_state.face_gate = {"passed": False, "distance": None, "error": None, "shot": None}
if "damage" not in st.session_state:
    st.session_state.damage = {"checked": False, "per_side": {}, "any_damage": None, "threshold": damage_threshold}

# =============== YOLO PREDICTION ===============
def yolo_predict(image: Image.Image):
    """
    Runs YOLO on a webcam PIL image.
    Feeds a NumPy BGR array (supported by Ultralytics).
    Returns: detected_ids (set[int]), annotated_png_bytes (bytes|None)
    """
    if yolo_model is None:
        st.warning("YOLO model is not loaded.")
        return set(), None

    # PIL -> NumPy (BGR)
    arr_rgb = np.array(image.convert("RGB"))
    arr_bgr = arr_rgb[:, :, ::-1].copy()

    try:
        preds = yolo_model.predict(source=arr_bgr, conf=conf_thres, imgsz=img_size, verbose=False)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return set(), None

    if not preds:
        return set(), None

    res = preds[0]
    detected_ids = set()
    try:
        if hasattr(res, "boxes") and res.boxes is not None and res.boxes.cls is not None:
            detected_ids = {int(i) for i in res.boxes.cls.tolist()}
    except Exception:
        pass

    annotated_bytes = None
    try:
        if hasattr(res, "plot"):
            plotted_bgr = res.plot()  # np.ndarray
            if plotted_bgr.ndim == 3 and plotted_bgr.shape[2] == 3:
                plotted_rgb = plotted_bgr[:, :, ::-1]
            else:
                plotted_rgb = plotted_bgr
            img = Image.fromarray(plotted_rgb)
            out = io.BytesIO()
            img.save(out, format="PNG")
            annotated_bytes = out.getvalue()
    except Exception:
        pass

    return detected_ids, annotated_bytes

# =============== DAMAGE INFERENCE ===============
def run_damage_on_images(images: Dict[str, bytes], threshold: float) -> Dict[str, Dict[str, Any]]:
    """
    images: dict side_key -> image bytes (PNG/JPEG)
    returns: per_side dict with:
        { side_key: {
            "label": str|None,
            "score": float|None,
            "is_damaged": bool|None
        }}
    """
    pipe = load_damage_pipe()
    results: Dict[str, Dict[str, Any]] = {}
    if pipe is None:
        for k in images.keys():
            results[k] = {"label": None, "score": None, "is_damaged": None}
        return results

    for side, b in images.items():
        if not b:
            results[side] = {"label": None, "score": None, "is_damaged": None}
            continue
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            out = pipe(img)
            # pipeline returns list of dicts sorted by score
            top = out[0] if isinstance(out, list) and len(out) > 0 else None
            if top:
                label = str(top.get("label", "")).lower()
                score = float(top.get("score", 0.0))
                is_damage_label = ("damag" in label)  # match "damage"/"damaged"
                is_damaged = bool(is_damage_label and score >= threshold)
                results[side] = {"label": label, "score": score, "is_damaged": is_damaged}
            else:
                results[side] = {"label": None, "score": None, "is_damaged": None}
        except Exception as e:
            results[side] = {"label": None, "score": None, "is_damaged": None}
            st.warning(f"Damage model failed on {side}: {e}")
    return results

# =============== UI PANELS ===============
def gate_panel():
    st.header("Face Verification Gate (Webcam)")
    st.write("Сделайте снимок с **двумя лицами** в кадре: ваше лицо и фото на документе.")

    cam = st.camera_input("Снимок для сверки лиц")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, caption="Снимок (оригинал)", use_container_width=True)

        matched, distance, err, num_faces = verify_two_faces_in_one(img)
        st.session_state.face_gate.update(
            {"passed": matched, "distance": distance, "error": err, "shot": cam.getvalue()}
        )

        if err:
            st.error(f"{err} (обнаружено лиц: {num_faces})")
        else:
            st.success(f"Сверка выполнена: {'Совпадение ✅' if matched else 'Не совпало ❌'} | distance={distance:.4f}")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.session_state.face_gate["passed"]:
            if st.button("Next → Front"):
                st.session_state.step = 1
                st.rerun()
        else:
            st.info("Сначала необходимо успешное совпадение (ровно 2 лица и verified=True).")

def side_panel(side_key: str):
    label = SIDE_LABELS[side_key]
    st.header(f"{label} (Webcam)")
    st.caption("Сделайте четкий кадр этой стороны авто.")

    cam = st.camera_input(f"Камера — {label}")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, caption=f"{label} — Оригинал", use_container_width=True)

        detected_ids, annotated = yolo_predict(img)
        side_true = len(SIDE_RULES[side_key].intersection(detected_ids)) >= 0

        st.session_state.sides[side_key] = {
            "true": side_true, "detected_ids": detected_ids,
            "shot": cam.getvalue(), "annotated": annotated
        }

        if annotated:
            st.image(annotated, caption=f"{label} — Аннотация", use_container_width=True)

        st.metric(f"{label} side", "TRUE ✅" if side_true else "FALSE ❌")
        if detected_ids:
            st.write("Detected IDs:", sorted(detected_ids))
            st.write("Detected names:", ", ".join(ids_to_names(detected_ids)))

    # Nav buttons
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("← Back"):
            st.session_state.step -= 1
            st.rerun()
    with cols[1]:
        can_go_next = st.session_state.sides[side_key]["true"] is True
        next_label = {
            "front": "Next → Left",
            "left": "Next → Back",
            "back": "Next → Right",
            "right": "Finish → Summary",
        }[side_key]
        if can_go_next:
            if st.button(next_label):
                if side_key == "right":
                    st.session_state.step = 5  # summaxsry
                else:
                    st.session_state.step += 1
                st.rerun()
        else:
            st.info("Для перехода далее этот ракурс должен быть TRUE (обнаружен хотя бы один ожидаемый класс).")

def summary_panel():
    st.header("Summary")
    for k in SIDE_ORDER:
        v = st.session_state.sides[k]
        label = SIDE_LABELS[k]
        status = "⏳ not checked"
        if v["true"] is True:
            status = "TRUE ✅"
        elif v["true"] is False:
            status = "FALSE ❌"
        st.write(f"**{label}** — {status}")
        if v["detected_ids"]:
            st.caption("Detected: " + ", ".join(ids_to_names(v["detected_ids"])))

    st.markdown("---")
    st.subheader("Damage Detection (all sides)")

    # Only run once per visit or if threshold changed
    threshold_changed = st.session_state.damage.get("threshold") != damage_threshold
    need_run = (not st.session_state.damage["checked"]) or threshold_changed

    if need_run:
        # Collect available side shots
        imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}
        per_side = run_damage_on_images(imgs, damage_threshold)
        # Persist
        st.session_state.damage["per_side"] = per_side
        st.session_state.damage["checked"] = True
        st.session_state.damage["threshold"] = damage_threshold

    per_side = st.session_state.damage["per_side"]
    any_damage = False
    for k in SIDE_ORDER:
        label = SIDE_LABELS[k]
        r = per_side.get(k, {})
        lab = r.get("label")
        score = r.get("score")
        is_dmg = r.get("is_damaged")
        if is_dmg:
            any_damage = True
        line = f"**{label}** — "
        if is_dmg is None:
            line += "_no image or model unavailable_"
        else:
            if lab is not None and score is not None:
                line += f"`{lab}` (score={score:.3f}) → {'DAMAGED 🔴' if is_dmg else 'OK 🟢'}"
            else:
                line += "_no prediction_"
        st.write(line)

    st.markdown("---")
    if any_damage:
        st.error("Result: **Not acceptable for taxi driver** (damage detected on at least one side).")
    else:
        st.success("Result: **Acceptable for taxi driver** (no damage detected above threshold).")

    st.caption("You can adjust the damage threshold in the sidebar and revisit this summary.")

# =============== ROUTER ===============
st.title("Webcam Car Side Verifier (YOLOv11-seg) + Face Gate + Damage Detection")

step = st.session_state.step
if step == 0:
    gate_panel()
elif step in (1, 2, 3, 4):
    side_key = SIDE_ORDER[step - 1]
    side_panel(side_key)
elif step == 5:
    summary_panel()
else:
    st.session_state.step = 0
    st.rerun()

# import io
# from typing import Tuple, Optional, Dict, Any

# import streamlit as st
# from PIL import Image
# import numpy as np

# # =============== PAGE CONFIG ===============
# st.set_page_config(page_title="Webcam Car Side Verifier + Face Gate + Damage + Cleanliness", layout="wide")

# # =============== CLASS MAPS / RULES ===============
# CLASS_MAP = {
#     0: "back_bumper",
#     1: "hood",
#     2: "back_glass",
#     3: "back_left_door",
#     4: "trunk",
#     5: "back_light",
#     6: "back_right_door",
#     7: "roof",
#     8: "front_bumper",
#     9: "grille",
#     10: "front_glass",
#     11: "front_left_door",
#     12: "bonnet",
#     13: "front_light",
#     14: "front_right_door",
#     15: "wheel",
#     16: "license_plate",
#     17: "left_mirror",
#     18: "spoiler",
#     19: "right_mirror",
# }

# SIDE_RULES = {
#     "front": {8, 10, 13},   # front_bumper, front_glass, front_light
#     "left":  {11, 3, 17},   # front_left_door, back_left_door, left_mirror
#     "back":  {0, 2, 5},     # back_bumper, back_glass, back_light
#     "right": {6, 14, 19},   # back_right_door, front_right_door, right_mirror
# }

# SIDE_ORDER = ["front", "left", "back", "right"]
# SIDE_LABELS = {"front": "Front", "left": "Left", "back": "Back", "right": "Right"}

# # === APP STEPS ===
# STEP_MENU = -1
# STEP_FACE = 0
# STEP_FRONT, STEP_LEFT, STEP_BACK, STEP_RIGHT = 1, 2, 3, 4
# STEP_SUMMARY = 5
# STEP_DAMAGE_ONLY = 6
# STEP_CLEAN_ONLY = 7
# STEP_REGISTER = 8


# def ids_to_names(ids):
#     return sorted({CLASS_MAP.get(int(i), f"class_{int(i)}") for i in ids})


# # =============== UTIL: PIL <-> CV2 ===============
# def pil_to_cv(image: Image.Image):
#     arr = np.array(image.convert("RGB"))
#     return arr[:, :, ::-1].copy()


# # =============== DEEPFACE GATE ===============
# # pip install deepface retina-face opencv-python
# from deepface import DeepFace

# def verify_two_faces_in_one(image: Image.Image) -> Tuple[bool, Optional[float], Optional[str], int]:
#     try:
#         arr_bgr = pil_to_cv(image)
#         faces = DeepFace.extract_faces(
#             img_path=arr_bgr,
#             detector_backend="retinaface",
#             enforce_detection=False
#         )
#         num_faces = len(faces)

#         if num_faces < 2:
#             return False, None, (
#                 "На снимке найдено меньше двух лиц. Держите документ ближе к камере и следите, "
#                 "чтобы портрет на документе был хорошо виден."
#             ), num_faces

#         if num_faces > 2:
#             return False, None, (
#                 "На снимке обнаружено больше двух лиц. Уберите лишних людей из кадра."
#             ), num_faces

#         f1 = faces[0]["face"]
#         f2 = faces[1]["face"]
#         result = DeepFace.verify(
#             img1_path=f1,
#             img2_path=f2,
#             model_name="VGG-Face",
#             detector_backend="opencv",
#             enforce_detection=False
#         )
#         matched = bool(result.get("verified", False))
#         distance = float(result.get("distance", 0.0))
#         return matched, distance, None, num_faces

#     except Exception as e:
#         return False, None, f"Ошибка при анализе лиц: {e}", 0


# # =============== MODEL LOADING (YOLO DETECT) ===============
# @st.cache_resource(show_spinner=True)
# def load_yolo():
#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         st.error("Ultralytics not installed. Run: `pip install ultralytics`")
#         return None
#     # ❗️Укажи путь к детекционным весам (не segment): например, yolo11n.pt или твой best.pt из runs/detect
#     weights_path = "/hdd/home/ABaglanova/indrive_hack/runs/detect/train/weights/best.pt"
#     try:
#         model = YOLO(weights_path)
#         try:
#             if hasattr(model, "names"):
#                 names = {int(i): str(n) for i, n in model.names.items()}
#                 if names:
#                     global CLASS_MAP
#                     CLASS_MAP = names
#         except Exception:
#             pass
#         return model
#     except Exception as e:
#         st.error(f"Failed to load YOLO weights at {weights_path}\n\n{e}")
#         return None

# yolo_model = load_yolo()


# # =============== DAMAGE PIPELINE (HF Transformers) ===============
# @st.cache_resource(show_spinner=True)
# def load_damage_pipe():
#     try:
#         from transformers import pipeline
#     except Exception:
#         st.error("Transformers not installed. Run: `pip install transformers pillow`")
#         return None
#     try:
#         return pipeline("image-classification", model="beingamit99/car_damage_detection")
#     except Exception as e:
#         st.error(f"Failed to load damage model: {e}")
#         return None


# # =============== CLEANLINESS CLASSIFIER (VGG16) ===============
# @st.cache_resource(show_spinner=True)
# def load_cleanliness_model():
#     try:
#         import torch
#         import torch.nn as nn
#         from torchvision import models
#     except ImportError:
#         st.error("PyTorch/torchvision not installed. Run: `pip install torch torchvision`")
#         return None

#     ckpt_path = "/hdd/home/ABaglanova/indrive_hack/vgg16_dirty_clean_best.pth"
#     try:
#         model = models.vgg16(weights=None)
#         model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
#         import torch
#         state = torch.load(ckpt_path, map_location="cpu")
#         if isinstance(state, dict) and "state_dict" in state:
#             state = state["state_dict"]
#         state = {k.replace("module.", ""): v for k, v in state.items()}
#         model.load_state_dict(state, strict=False)
#         model.eval()
#         return model
#     except Exception as e:
#         st.error(f"Failed to load cleanliness model: {e}")
#         return None


# def cleanliness_predict_batch(images: Dict[str, bytes], threshold: float) -> Dict[str, Dict[str, Any]]:
#     model = load_cleanliness_model()
#     results: Dict[str, Dict[str, Any]] = {}
#     if model is None:
#         for k in images.keys():
#             results[k] = {"probs": None, "is_dirty": None}
#         return results

#     import torch
#     from torchvision import transforms
#     tfm = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])

#     with torch.no_grad():
#         for side, b in images.items():
#             try:
#                 if not b:
#                     results[side] = {"probs": None, "is_dirty": None}
#                     continue
#                 img = Image.open(io.BytesIO(b)).convert("RGB")
#                 x = tfm(img).unsqueeze(0)
#                 logits = model(x)
#                 probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
#                 prob_clean, prob_dirty = float(probs[0]), float(probs[1])  # 0=clean, 1=dirty
#                 is_dirty = prob_dirty >= threshold
#                 results[side] = {"probs": {"clean": prob_clean, "dirty": prob_dirty}, "is_dirty": is_dirty}
#             except Exception as e:
#                 st.warning(f"Cleanliness model failed on {side}: {e}")
#                 results[side] = {"probs": None, "is_dirty": None}
#     return results


# # =============== SIDEBAR CONTROLS ===============
# with st.sidebar:
#     st.markdown("### YOLO Inference Settings")
#     conf_thres = st.slider("YOLO confidence", 0.05, 0.75, 0.25, 0.01)
#     img_size   = st.slider("YOLO image size", 320, 1280, 640, 32)
#     st.markdown("---")
#     st.markdown("### Damage Detection Settings")
#     damage_threshold = st.slider(
#         "Damage confidence threshold",
#         0.10, 0.95, 0.50, 0.01,
#         help="If the top damage score ≥ this value on any side, the car is NOT acceptable for taxi."
#     )
#     st.caption("Model: beingamit99/car_damage_detection")
#     st.markdown("---")
#     st.markdown("### Cleanliness (VGG16)")
#     cleanliness_threshold = st.slider(
#         "DIRTY probability threshold",
#         0.10, 0.95, 0.50, 0.01,
#         help="If P(dirty) ≥ this value on any side, считаем сторону грязной."
#     )
#     st.caption("Weights: vgg16_dirty_clean_best.pth")


# # =============== SESSION STATE ===============
# if "step" not in st.session_state:
#     st.session_state.step = STEP_MENU
# if "sides" not in st.session_state:
#     st.session_state.sides = {
#         k: {"true": None, "detected_ids": set(), "shot": None, "annotated": None}
#         for k in SIDE_ORDER
#     }
# if "face_gate" not in st.session_state:
#     st.session_state.face_gate = {"passed": False, "distance": None, "error": None, "shot": None}
# if "damage" not in st.session_state:
#     st.session_state.damage = {"checked": False, "per_side": {}, "any_damage": None, "threshold": damage_threshold}
# if "cleanliness" not in st.session_state:
#     st.session_state.cleanliness = {"checked": False, "per_side": {}, "any_dirty": None, "threshold": cleanliness_threshold}
# if "user" not in st.session_state:
#     st.session_state.user = {}


# # =============== YOLO PREDICTION (DETECT ONLY) ===============
# def yolo_predict(image: Image.Image):
#     if yolo_model is None:
#         st.warning("YOLO model is not loaded.")
#         return set(), None

#     arr_rgb = np.array(image.convert("RGB"))
#     arr_bgr = arr_rgb[:, :, ::-1].copy()

#     try:
#         preds = yolo_model.predict(source=arr_bgr, conf=conf_thres, imgsz=img_size, verbose=False)
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         return set(), None

#     if not preds:
#         return set(), None

#     res = preds[0]
#     detected_ids = set()
#     try:
#         if hasattr(res, "boxes") and res.boxes is not None and res.boxes.cls is not None:
#             detected_ids = {int(i) for i in res.boxes.cls.tolist()}
#     except Exception:
#         pass

#     annotated_bytes = None
#     try:
#         if hasattr(res, "plot"):  # отрисуем только боксы; сегментации у нас нет
#             plotted_bgr = res.plot()
#             plotted_rgb = plotted_bgr[:, :, ::-1] if plotted_bgr.ndim == 3 and plotted_bgr.shape[2] == 3 else plotted_bgr
#             img = Image.fromarray(plotted_rgb)
#             out = io.BytesIO()
#             img.save(out, format="PNG")
#             annotated_bytes = out.getvalue()
#     except Exception:
#         pass

#     return detected_ids, annotated_bytes


# # =============== DAMAGE INFERENCE ===============
# def run_damage_on_images(images: Dict[str, bytes], threshold: float) -> Dict[str, Dict[str, Any]]:
#     pipe = load_damage_pipe()
#     results: Dict[str, Dict[str, Any]] = {}
#     if pipe is None:
#         for k in images.keys():
#             results[k] = {"label": None, "score": None, "is_damaged": None}
#         return results

#     for side, b in images.items():
#         if not b:
#             results[side] = {"label": None, "score": None, "is_damaged": None}
#             continue
#         try:
#             img = Image.open(io.BytesIO(b)).convert("RGB")
#             out = pipe(img)
#             top = out[0] if isinstance(out, list) and len(out) > 0 else None
#             if top:
#                 label = str(top.get("label", "")).lower()
#                 score = float(top.get("score", 0.0))
#                 is_damage_label = ("damag" in label)
#                 is_damaged = bool(is_damage_label and score >= threshold)
#                 results[side] = {"label": label, "score": score, "is_damaged": is_damaged}
#             else:
#                 results[side] = {"label": None, "score": None, "is_damaged": None}
#         except Exception as e:
#             results[side] = {"label": None, "score": None, "is_damaged": None}
#             st.warning(f"Damage model failed on {side}: {e}")
#     return results


# # =============== UI PANELS ===============
# def gate_panel():
#     st.header("Face Verification Gate (Webcam)")
#     st.write("Сделайте снимок с **двумя лицами** в кадре: ваше лицо и фото на документе.")

#     cam = st.camera_input("Снимок для сверки лиц")
#     if cam:
#         img = Image.open(cam).convert("RGB")
#         st.image(img, caption="Снимок (оригинал)", use_container_width=True)

#         matched, distance, err, num_faces = verify_two_faces_in_one(img)
#         st.session_state.face_gate.update(
#             {"passed": matched, "distance": distance, "error": err, "shot": cam.getvalue()}
#         )

#         if err:
#             st.error(f"{err} (обнаружено лиц: {num_faces})")
#         else:
#             st.success(f"Сверка выполнена: {'Совпадение ✅' if matched else 'Не совпало ❌'} | distance={distance:.4f}")

#     col1, _ = st.columns([1, 2])
#     with col1:
#         if st.session_state.face_gate["passed"]:
#             if st.button("Next → Front"):
#                 st.session_state.step = STEP_FRONT
#                 st.rerun()
#         else:
#             st.info("Сначала необходимо успешное совпадение (ровно 2 лица и verified=True).")

#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# def side_panel(side_key: str):
#     label = SIDE_LABELS[side_key]
#     st.header(f"{label} (Webcam)")
#     st.caption("Сделайте четкий кадр этой стороны авто.")

#     cam = st.camera_input(f"Камера — {label}")
#     if cam:
#         img = Image.open(cam).convert("RGB")
#         st.image(img, caption=f"{label} — Оригинал", use_container_width=True)

#         detected_ids, annotated = yolo_predict(img)
#         side_true = len(SIDE_RULES[side_key].intersection(detected_ids)) > 0

#         st.session_state.sides[side_key] = {
#             "true": side_true, "detected_ids": detected_ids,
#             "shot": cam.getvalue(), "annotated": annotated
#         }

#         if annotated:
#             st.image(annotated, caption=f"{label} — Аннотация", use_container_width=True)

#         st.metric(f"{label} side", "TRUE ✅" if side_true else "FALSE ❌")
#         if detected_ids:
#             st.write("Detected IDs:", sorted(detected_ids))
#             st.write("Detected names:", ", ".join(ids_to_names(detected_ids)))

#     cols = st.columns([1, 1, 6])
#     with cols[0]:
#         if st.button("← Back"):
#             st.session_state.step -= 1
#             st.rerun()
#     with cols[1]:
#         can_go_next = st.session_state.sides[side_key]["true"] is True
#         next_label = {
#             "front": "Next → Left",
#             "left": "Next → Back",
#             "back": "Next → Right",
#             "right": "Finish → Summary",
#         }[side_key]
#         if can_go_next:
#             if st.button(next_label):
#                 if side_key == "right":
#                     st.session_state.step = STEP_SUMMARY
#                 else:
#                     st.session_state.step += 1
#                 st.rerun()
#         else:
#             st.info("Для перехода далее этот ракурс должен быть TRUE (обнаружен хотя бы один ожидаемый класс).")

#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# def summary_panel():
#     st.header("Summary")
#     for k in SIDE_ORDER:
#         v = st.session_state.sides[k]
#         label = SIDE_LABELS[k]
#         status = "⏳ not checked"
#         if v["true"] is True:
#             status = "TRUE ✅"
#         elif v["true"] is False:
#             status = "FALSE ❌"
#         st.write(f"**{label}** — {status}")
#         if v["detected_ids"]:
#             st.caption("Detected: " + ", ".join(ids_to_names(v["detected_ids"])))

#     st.markdown("---")
#     st.subheader("Damage Detection (all sides)")

#     threshold_changed = st.session_state.damage.get("threshold") != damage_threshold
#     need_run = (not st.session_state.damage["checked"]) or threshold_changed

#     if need_run:
#         imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}
#         per_side = run_damage_on_images(imgs, damage_threshold)
#         st.session_state.damage["per_side"] = per_side
#         st.session_state.damage["checked"] = True
#         st.session_state.damage["threshold"] = damage_threshold

#     per_side = st.session_state.damage["per_side"]
#     any_damage = False
#     for k in SIDE_ORDER:
#         label = SIDE_LABELS[k]
#         r = per_side.get(k, {})
#         lab = r.get("label")
#         score = r.get("score")
#         is_dmg = r.get("is_damaged")
#         if is_dmg:
#             any_damage = True
#         line = f"**{label}** — "
#         if is_dmg is None:
#             line += "_no image or model unavailable_"
#         else:
#             if lab is not None and score is not None:
#                 line += f"`{lab}` (score={score:.3f}) → {'DAMAGED 🔴' if is_dmg else 'OK 🟢'}"
#             else:
#                 line += "_no prediction_"
#         st.write(line)

#     st.markdown("---")
#     st.subheader("Cleanliness (VGG16 dirty/clean)")

#     cln_threshold_changed = st.session_state.cleanliness.get("threshold") != cleanliness_threshold
#     cln_need_run = (not st.session_state.cleanliness["checked"]) or cln_threshold_changed

#     if cln_need_run:
#         imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}
#         cln_per_side = cleanliness_predict_batch(imgs, cleanliness_threshold)
#         st.session_state.cleanliness["per_side"] = cln_per_side
#         st.session_state.cleanliness["checked"] = True
#         st.session_state.cleanliness["threshold"] = cleanliness_threshold

#     cln_per_side = st.session_state.cleanliness["per_side"]
#     any_dirty = False
#     for k in SIDE_ORDER:
#         label = SIDE_LABELS[k]
#         r = cln_per_side.get(k, {})
#         probs = r.get("probs")
#         is_dirty = r.get("is_dirty")
#         if is_dirty:
#             any_dirty = True
#         line = f"**{label}** — "
#         if probs is None or is_dirty is None:
#             line += "_no image or model unavailable_"
#         else:
#             pc = probs["clean"]; pd = probs["dirty"]
#             line += f"P(clean)={pc:.3f}, P(dirty)={pd:.3f} → {'DIRTY 🟠' if is_dirty else 'CLEAN 🟢'}"
#         st.write(line)

#     st.caption("You can adjust the damage & cleanliness thresholds in the sidebar and revisit this summary.")

#     st.markdown("---")
#     if any_damage:
#         st.error("Result: **Not acceptable for taxi driver** (damage detected on at least one side).")
#     else:
#         st.success("Result: **Acceptable for taxi driver** (no damage detected above threshold).")

#     if any_dirty:
#         st.warning("Cleanliness: обнаружены **грязные** стороны по порогу вероятности.")
#     else:
#         st.info("Cleanliness: все заснятые стороны **чистые** по текущему порогу.")

#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# # ======= MENU + QUICK PANELS + REGISTRATION =======
# def menu_panel():
#     st.title("Панель действий")
#     st.caption("Выберите, что сделать:")

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if st.button("🔧 Проверить повреждения", use_container_width=True):
#             st.session_state.step = STEP_DAMAGE_ONLY
#             st.rerun()
#     with c2:
#         if st.button("🧼 Проверить грязь", use_container_width=True):
#             st.session_state.step = STEP_CLEAN_ONLY
#             st.rerun()
#     with c3:
#         if st.button("🪪 Регистрация пользователя", use_container_width=True):
#             st.session_state.step = STEP_REGISTER
#             st.rerun()

#     st.markdown("---")
#     st.caption("Или запустите стандартный процесс: Face Gate → Стороны → Сводка")
#     if st.button("▶️ Запустить стандартный процесс"):
#         st.session_state.step = STEP_FACE
#         st.rerun()


# def damage_only_panel():
#     st.header("Проверка на повреждения (быстрый режим)")
#     imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}
#     if not imgs:
#         st.info("Пока нет снимков. Сделайте кадры на шагах Front/Left/Back/Right или загрузите ниже.")
#         up_front = st.file_uploader("Front", type=["jpg","jpeg","png"], key="up_front_dmg")
#         up_left  = st.file_uploader("Left",  type=["jpg","jpeg","png"], key="up_left_dmg")
#         up_back  = st.file_uploader("Back",  type=["jpg","jpeg","png"], key="up_back_dmg")
#         up_right = st.file_uploader("Right", type=["jpg","jpeg","png"], key="up_right_dmg")
#         uploads = {"front": up_front, "left": up_left, "back": up_back, "right": up_right}
#         for k, f in uploads.items():
#             if f is not None:
#                 st.session_state.sides[k]["shot"] = f.read()
#         imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}

#     if imgs:
#         results = run_damage_on_images(imgs, damage_threshold)
#         any_damage = False
#         for k in SIDE_ORDER:
#             r = results.get(k, {})
#             label, score, is_dmg = r.get("label"), r.get("score"), r.get("is_damaged")
#             line = f"**{SIDE_LABELS[k]}** — "
#             if is_dmg is None:
#                 line += "_no image_"
#             else:
#                 if label is not None and score is not None:
#                     line += f"`{label}` (score={score:.3f}) → {'DAMAGED 🔴' if is_dmg else 'OK 🟢'}"
#                 else:
#                     line += "_no prediction_"
#             st.write(line)
#             any_damage = any_damage or bool(is_dmg)
#         st.markdown("---")
#         st.error("Итог: Обнаружены повреждения — НЕ допускается к выезду.") if any_damage \
#             else st.success("Итог: Повреждений выше порога не обнаружено — ОК.")

#     st.markdown("---")
#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# def cleanliness_only_panel():
#     st.header("Проверка на грязь (быстрый режим)")
#     imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}
#     if not imgs:
#         st.info("Пока нет снимков. Сделайте кадры или загрузите ниже.")
#         up_front = st.file_uploader("Front", type=["jpg","jpeg","png"], key="up_front_cln")
#         up_left  = st.file_uploader("Left",  type=["jpg","jpeg","png"], key="up_left_cln")
#         up_back  = st.file_uploader("Back",  type=["jpg","jpeg","png"], key="up_back_cln")
#         up_right = st.file_uploader("Right", type=["jpg","jpeg","png"], key="up_right_cln")
#         uploads = {"front": up_front, "left": up_left, "back": up_back, "right": up_right}
#         for k, f in uploads.items():
#             if f is not None:
#                 st.session_state.sides[k]["shot"] = f.read()
#         imgs = {k: st.session_state.sides[k]["shot"] for k in SIDE_ORDER if st.session_state.sides[k]["shot"]}

#     if imgs:
#         cln = cleanliness_predict_batch(imgs, cleanliness_threshold)
#         any_dirty = False
#         for k in SIDE_ORDER:
#             r = cln.get(k, {})
#             probs, is_dirty = r.get("probs"), r.get("is_dirty")
#             line = f"**{SIDE_LABELS[k]}** — "
#             if probs is None or is_dirty is None:
#                 line += "_no image_"
#             else:
#                 pc, pd = probs["clean"], probs["dirty"]
#                 line += f"P(clean)={pc:.3f}, P(dirty)={pd:.3f} → {'DIRTY 🟠' if is_dirty else 'CLEAN 🟢'}"
#             st.write(line)
#             any_dirty = any_dirty or bool(is_dirty)
#         st.markdown("---")
#         st.warning("Итог: обнаружены грязные стороны.") if any_dirty \
#             else st.info("Итог: грязи по порогу не выявлено.")

#     st.markdown("---")
#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# def registration_panel():
#     st.header("Регистрация пользователя")
#     with st.form("register_form", clear_on_submit=False):
#         col1, col2 = st.columns(2)
#         with col1:
#             full_name = st.text_input("ФИО *")
#             phone     = st.text_input("Телефон *", placeholder="+7 ...")
#             email     = st.text_input("Email")
#         with col2:
#             car_make  = st.text_input("Марка авто *", placeholder="Toyota")
#             car_model = st.text_input("Модель авто *", placeholder="Camry")
#             plate     = st.text_input("Гос. номер *")
#         selfie = st.file_uploader("Загрузите селфи с документом (опционально)", type=["jpg","jpeg","png"])
#         submitted = st.form_submit_button("Сохранить регистрацию")

#     if submitted:
#         missing = [n for n, v in {
#             "ФИО": full_name, "Телефон": phone, "Марка": car_make,
#             "Модель": car_model, "Гос. номер": plate
#         }.items() if not v]
#         if missing:
#             st.error("Заполните обязательные поля: " + ", ".join(missing))
#         else:
#             st.session_state.user = {
#                 "full_name": full_name, "phone": phone, "email": email,
#                 "car_make": car_make, "car_model": car_model, "plate": plate,
#                 "selfie_present": selfie is not None
#             }
#             st.success("Регистрация сохранена.")
#             if selfie is not None:
#                 st.image(Image.open(selfie), caption="Загруженное фото", use_container_width=True)

#     if st.session_state.get("user"):
#         st.markdown("---")
#         st.subheader("Текущая карточка")
#         u = st.session_state.user
#         st.write(f"**{u['full_name']}** — {u['phone']}")
#         st.write(f"Авто: {u['car_make']} {u['car_model']}, номер {u['plate']}")
#         if u.get("email"):
#             st.write(f"Email: {u['email']}")
#         st.write("Фото приложено" if u["selfie_present"] else "Фото не приложено")

#     st.markdown("---")
#     if st.button("⬅️ Назад в меню"):
#         st.session_state.step = STEP_MENU
#         st.rerun()


# # =============== ROUTER ===============
# st.title("Webcam Car Side Verifier (YOLO Detect) + Face Gate + Damage + Cleanliness")

# step = st.session_state.step
# if step == STEP_MENU:
#     menu_panel()
# elif step == STEP_FACE:
#     gate_panel()
# elif step in (STEP_FRONT, STEP_LEFT, STEP_BACK, STEP_RIGHT):
#     side_key = SIDE_ORDER[step - 1]
#     side_panel(side_key)
# elif step == STEP_SUMMARY:
#     summary_panel()
# elif step == STEP_DAMAGE_ONLY:
#     damage_only_panel()
# elif step == STEP_CLEAN_ONLY:
#     cleanliness_only_panel()
# elif step == STEP_REGISTER:
#     registration_panel()
# else:
#     st.session_state.step = STEP_MENU
#     st.rerun()
