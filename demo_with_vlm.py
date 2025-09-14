# pip install -U "transformers>=4.44" "accelerate>=0.34" pillow torchvision timm sentencepiece safetensors streamlit python-dotenv deepface
import base64
import io
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from deepface import DeepFace
import streamlit as st

# ==== Qwen2.5-VL ====
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- (опциональный) фолбэк, если нет qwen_vl_utils.process_vision_info ---
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    def process_vision_info(messages):
        """Минимальный фолбэк: вытащить PIL-изображения из messages.
        Возвращает (image_inputs, video_inputs) как ожидает процессор Qwen.
        """
        image_inputs = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image_inputs.append(item["image"])  # PIL.Image или путь
        return image_inputs, None

# ---------- Утилиты ----------
def file_to_pil(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

def pil_to_cv(img: Image.Image):
    # DeepFace принимает numpy ƒçBGR или путь; используем RGB->BGR
    arr = np.array(img)[:, :, ::-1]
    return arr

def pil_to_data_url(img: Image.Image, format="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"
ç
# ---------- Qwen2.5-VL загрузка (кэшируем, чтобы не грузить каждый раз) ----------
@st.cache_resource(show_spinner=True)
def load_qwen():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return processor, model

# ---------- Сверка двух лиц в одном фото через DeepFace ----------
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

# ---------- Визуальный анализ авто через Qwen2.5-VL ----------
SYSTEM_PROMPT = (
    "Ты — аккуратный инспектор авто. Твоя задача: определить ракурс (front|side-left|side-right|rear), "
    "чистоту (clean: true/false), есть ли видимые повреждения (damage: true/false), и краткие заметки. "
    "Отвечай ТОЛЬКО JSON по схеме: "
    '{"angle":"front|side-left|side-right|rear","clean":true/false,"damage":true/false,"notes":"строка"}'
)

def analyze_car_image(img: Image.Image) -> dict:
    """Тот же сценарий инференса, но через Qwen2.5-VL; без выдумок, только фото."""
    processor, model = load_qwen()

    user_prompt = (
        "Проанализируй фото автомобиля. Определи ракурс (front/side-left/side-right/rear), "
        "чистоту (clean) и наличие видимых повреждений (damage). Верни только JSON."
    )

    # строим messages строго под Qwen как в их примере
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt},
                                     {"type": "image", "image": img}]},
    ]

    # 1) строковый промпт
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # 2) визуальные входы
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) пакуем
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )

    # 4) на устройство модели
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 5) генерация
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,   # детерминистичнее для прод-логики
        )

    # берём только новые токены
    new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    text_out = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # попытка выдрать JSON
    try:
        first_brace = text_out.find("{")
        last_brace = text_out.rfind("}")
        if first_brace != -1 and last_brace != -1:
            text_out = text_out[first_brace:last_brace+1]
        parsed = json.loads(text_out)
        # нормализуем типы/ключи
        return {
            "angle": str(parsed.get("angle", "unknown")),
            "clean": bool(parsed.get("clean", False)),
            "damage": bool(parsed.get("damage", True)),
            "notes": str(parsed.get("notes", "")),
        }
    except Exception:
        return {"angle": "unknown", "clean": False, "damage": True, "notes": "parse_error"}

# ---------- Стейт и модели данных ----------
@dataclass
class PhotoCheck:
    captured: bool = False
    passed: Optional[bool] = None
    angle_expected: Optional[str] = None
    angle_detected: Optional[str] = None
    clean: Optional[bool] = None
    damage: Optional[bool] = None
    notes: Optional[str] = None
    preview: Optional[bytes] = None  # для отображения

def init_state():
    if "step" not in st.session_state:
        st.session_state.step = "welcome"  # welcome -> selfie_doc -> car_front -> car_side_left -> car_rear -> car_side_right -> summary

    if "selfie_doc" not in st.session_state:
        st.session_state.selfie_doc = None
    if "face_match" not in st.session_state:
        st.session_state.face_match = None

    if "checks" not in st.session_state:
        st.session_state.checks = {
            "front": PhotoCheck(angle_expected="front"),
            "side_left": PhotoCheck(angle_expected="side-left"),
            "rear": PhotoCheck(angle_expected="rear"),
            "side_right": PhotoCheck(angle_expected="side-right"),
        }

    # Счётчики для сброса камеры (уникальные key у st.camera_input)
    if "camera_attempts" not in st.session_state:
        st.session_state.camera_attempts = {
            "selfie_doc": 0,
            "front": 0,
            "side_left": 0,
            "rear": 0,
            "side_right": 0,
        }

def go(next_step: str):
    st.session_state.step = next_step
    st.rerun()

def reset_camera(step_key: str):
    st.session_state.camera_attempts[step_key] += 1
    if step_key == "selfie_doc":
        st.session_state.selfie_doc = None
        st.session_state.face_match = None
    else:
        chk: PhotoCheck = st.session_state.checks[step_key]
        chk.captured = False
        chk.preview = None
        chk.angle_detected = None
        chk.clean = None
        chk.damage = None
        chk.notes = None
        chk.passed = None
    st.rerun()

# ---------- UI ----------
st.set_page_config(page_title="Проверка поездки", page_icon="🚗", layout="centered")
st.title("🚗 Проверка поездки")
st.caption("Streamlit • DeepFace (1 фото) • Qwen2.5-VL • Camera reset")

init_state()

with st.sidebar:
    st.header("Шаги")
    steps = [
        ("welcome", "Старт"),
        ("selfie_doc", "Селфи с документом"),
        ("car_front", "Фото авто спереди"),
        ("car_side_left", "Фото авто слева"),
        ("car_rear", "Фото авто сзади"),
        ("car_side_right", "Фото авто справа"),
        ("summary", "Итог"),
    ]
    for key, label in steps:
        flag = "✅" if (st.session_state.step == key or
                       (key == "summary" and st.session_state.step == "summary")) else ""
        st.write(f"{flag} {label}")

# -------- ХЕЛПЕР ДЛЯ ШАГОВ АВТО --------
def car_step_ui(step_key: str, title: str, hint: str):
    st.subheader(title)
    st.write(hint)

    cam_key = f"cam_{step_key}_{st.session_state.camera_attempts[step_key]}"
    shot = st.camera_input("Сделать фото", key=cam_key)

    chk: PhotoCheck = st.session_state.checks[step_key]
    if shot:
        img = file_to_pil(shot)
        st.image(img, caption="Предпросмотр", use_column_width=True)

        with st.spinner("Анализ фото (Qwen2.5-VL)…"):
            result = analyze_car_image(img)

        chk.captured = True
        chk.preview = shot.getvalue()
        chk.angle_detected = result.get("angle", "unknown")
        chk.clean = bool(result.get("clean", False))
        chk.damage = bool(result.get("damage", True))
        chk.notes = str(result.get("notes", ""))

        # Проходной критерий: верный ракурс + чисто + без повреждений
        chk.passed = (chk.angle_detected == chk.angle_expected) and chk.clean and (not chk.damage)

        cols = st.columns(3)
        cols[0].metric("Ожидаемый ракурс", chk.angle_expected or "-")
        cols[1].metric("Определён ракурс", chk.angle_detected or "-")
        cols[2].metric("Статус", "✅ OK" if chk.passed else "❌ Не пройдено")

        st.write(f"Чистота: **{chk.clean}**, Повреждения: **{chk.damage}**")
        if chk.notes:
            st.caption(f"Заметки: {chk.notes}")

        next_map = {
            "front": "car_side_left",
            "side_left": "car_rear",
            "rear": "car_side_right",
            "side_right": "summary",
        }

        btn_cols = st.columns(2)
        if btn_cols[0].button("Сделать другое фото 🔄"):
            reset_camera(step_key)
        if btn_cols[1].button("Дальше ➡️", type="primary"):
            go(next_map[step_key])
    else:
        st.button("Сбросить камеру 🔄", on_click=reset_camera, args=(step_key,))

# -------- РОУТИНГ ШАГОВ --------
if st.session_state.step == "welcome":
    st.subheader("Добро пожаловать!")
    st.write("Нажмите кнопку, чтобы начать процесс верификации и осмотра авто.")
    if st.button("Начать поездку ▶️", type="primary"):
        go("selfie_doc")

elif st.session_state.step == "selfie_doc":
    st.subheader("Селфи с документом (1 фото)")
    st.write("Держите документ так, чтобы было видно ваше лицо и портрет на документе. В кадре должны быть только вы.")

    cam_key = f"cam_selfie_{st.session_state.camera_attempts['selfie_doc']}"
    shot = st.camera_input("Сделать фото (селфи с документом)", key=cam_key)

    if shot:
        st.session_state.selfie_doc = file_to_pil(shot)
        st.image(st.session_state.selfie_doc, caption="Селфи с документом", use_column_width=True)

        with st.spinner("Ищем два лица и сверяем (DeepFace)…"):
            matched, distance, err, num_faces = verify_two_faces_in_one(st.session_state.selfie_doc)

        if err:
            st.error(err)
            st.button("Сделать другое фото 🔄", on_click=reset_camera, args=("selfie_doc",))
        else:
            st.session_state.face_match = {"matched": matched, "distance": distance, "num_faces": num_faces}
            if matched:
                st.success(f"Лица совпадают ✅ (distance={distance:.4f})")
                if st.button("Продолжить к авто ➡️", type="primary"):
                    go("car_front")
            else:
                st.error(f"Два лица найдены, но они не совпадают ❌ (distance={distance:.4f}). "
                         "Попробуйте другой ракурс/освещение.")
                st.button("Сделать другое фото 🔄", on_click=reset_camera, args=("selfie_doc",))
    else:
        st.button("Сбросить камеру 🔄", on_click=reset_camera, args=("selfie_doc",))

elif st.session_state.step == "car_front":
    car_step_ui(
        "front",
        "Фото авто спереди",
        "Сфотографируйте автомобиль строго спереди. Камеру держите на уровне фар."
    )

elif st.session_state.step == "car_side_left":
    car_step_ui(
        "side_left",
        "Фото авто слева (бок)",
        "Сделайте фото левого борта автомобиля полностью, колёса должны быть видны."
    )

elif st.session_state.step == "car_rear":
    car_step_ui(
        "rear",
        "Фото авто сзади",
        "Снимите авто строго сзади. Номер и фонари должны быть в кадре."
    )

elif st.session_state.step == "car_side_right":
    car_step_ui(
        "side_right",
        "Фото авто справа (бок)",
        "Сфотографируйте правый борт автомобиля полностью."
    )

elif st.session_state.step == "summary":
    st.subheader("Итоги проверки")

    # Итог по лицам
    face_ok = bool(st.session_state.face_match and st.session_state.face_match.get("matched"))
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Сверка лица (DeepFace):** " + ("✅ Успех" if face_ok else "❌ Не пройдено"))
        fm = st.session_state.face_match or {}
        if fm.get("distance") is not None:
            st.caption(f"distance = {fm['distance']:.4f}")
        if fm.get("num_faces") is not None:
            st.caption(f"обнаружено лиц: {fm['num_faces']}")
    with col_b:
        st.markdown("**Фото автомобиля:**")

    # Подробности по 4 ракурсам
    problems = []
    for k, label in [
        ("front", "Спереди"),
        ("side_left", "Слева"),
        ("rear", "Сзади"),
        ("side_right", "Справа"),
    ]:
        chk: PhotoCheck = st.session_state.checks[k]
        status = "✅ OK" if chk.passed else "❌"
        st.write(f"- **{label}:** {status} "
                 f"(ожидали: `{chk.angle_expected}`, определено: `{chk.angle_detected}`, "
                 f"clean={chk.clean}, damage={chk.damage})")
        if not chk.passed:
            if chk.angle_detected != chk.angle_expected:
                problems.append(f"{label}: неверный ракурс (ожидали {chk.angle_expected}, получили {chk.angle_detected})")
            if chk.clean is False:
                problems.append(f"{label}: авто выглядит грязным")
            if chk.damage is True:
                problems.append(f"{label}: обнаружены возможные повреждения")
            if chk.notes:
                problems.append(f"{label}: заметки — {chk.notes}")

    all_ok = face_ok and all(v.passed for v in st.session_state.checks.values())

    st.markdown("---")
    if all_ok:
        st.success("🎉 SUCCESS — все проверки пройдены!")
    else:
        st.error("Некоторые проверки не пройдены.")
        if problems:
            st.markdown("**Что не так:**")
            for p in problems:
                st.write(f"• {p}")

    # Кнопки
    col1, col2 = st.columns(2)
    if col1.button("Начать заново 🔄"):
        for key in ["step", "selfie_doc", "face_match", "checks", "camera_attempts"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    if col2.button("Завершить ✅"):
        st.success("Готово! Можете закрыть окно.")
