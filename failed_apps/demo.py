# demo.py
# -*- coding: utf-8 -*-
"""
Streamlit интерфейс с тремя кнопками в сайдбаре:
1) «найти повреждения» — инференс YOLOv8 по вашему весу
2) «найти загрязнения» — бинарная классификация VGG16 (clean/dirty)
3) «пройти регистрацию» — заглушка формы

Перед запуском:
  pip install streamlit ultralytics torch torchvision pillow

Запуск:
  streamlit run demo.py
"""

import os
from io import BytesIO
from typing import Optional, Union

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

# -----------------------------
# Пути из вашего запроса
# -----------------------------
YOLO_WEIGHTS = "trained.pt"
YOLO_TEST_IMG = "sample_images_for_tests/images.jpeg"
VGG16_WEIGHTS = "vgg16_dirty_clean_best.pth"

# -----------------------------
# Настройки страницы
# -----------------------------
st.set_page_config(page_title="Инспекция авто: повреждения и загрязнения", layout="wide")

# -----------------------------
# Кешируем загрузку моделей
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Не найден файл весов YOLO: {weights_path}")
    return YOLO(weights_path)

@st.cache_resource(show_spinner=False)
def build_vgg16_binary(weights_path: str):
    from torchvision import models
    model = models.vgg16(weights=None)
    in_features = model.classifier[-1].in_features
    new_classifier = list(model.classifier)
    new_classifier[-1] = torch.nn.Linear(in_features, 2)  # 2 класса: clean/dirty
    model.classifier = torch.nn.Sequential(*new_classifier)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Не найден файл весов VGG16: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def vgg_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -----------------------------
# Утилиты работы с изображениями
# -----------------------------
def open_image_any(source: Union[str, bytes, BytesIO, Image.Image]) -> Image.Image:
    """Открыть изображение и привести к RGB."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, BytesIO)):
        return Image.open(BytesIO(source if isinstance(source, bytes) else source.getvalue())).convert("RGB")
    if isinstance(source, str) and os.path.exists(source):
        return Image.open(source).convert("RGB")
    raise ValueError("Неподдерживаемый источник изображения")

def make_yolo_source(uploaded_file, fallback_path: Optional[str]) -> Optional[Union[Image.Image, str]]:
    """
    Вернуть тип, поддерживаемый Ultralytics:
      - если загружен файл -> PIL.Image
      - иначе -> путь к fallback-изображению, если существует
    """
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    return None

# -----------------------------
# UI: Sidebar
# -----------------------------
st.sidebar.title("Навигация")
btn_damage = st.sidebar.button("найти повреждения", use_container_width=True)
btn_dirty  = st.sidebar.button("найти загрязнения", use_container_width=True)
btn_reg    = st.sidebar.button("пройти регистрацию", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Demo • YOLOv8 + VGG16")

# -----------------------------
# Главная область
# -----------------------------
st.title("🚗 Инспекция автомобиля")
st.write("Выберите действие в левой панели или загрузите своё изображение ниже.")

with st.expander("Источник изображения (опционально)"):
    uploaded = st.file_uploader("Загрузите изображение (jpg/png)", type=["jpg", "jpeg", "png"])
    st.caption(f"Если файл не загружен — используется по умолчанию: {YOLO_TEST_IMG}")

# -----------------------------
# Кнопка 1: YOLO — повреждения
# -----------------------------
if btn_damage:
    st.subheader("🔎 Поиск повреждений (YOLOv8)")
    try:
        yolo_source = make_yolo_source(uploaded, YOLO_TEST_IMG)
        if yolo_source is None:
            st.error("Нет изображения: загрузите файл или проверьте путь по умолчанию.")
        else:
            model = load_yolo_model(YOLO_WEIGHTS)
            results = model(yolo_source)
            res = results[0]

            # Визуализация: res.plot() -> np.ndarray в BGR, конвертируем в RGB для PIL
            plotted_bgr = res.plot()
            img_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])
            st.image(img_rgb, caption="Результаты детекции (YOLOv8)", use_column_width=True)

            # Таблица предсказаний
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                import pandas as pd
                names = res.names if hasattr(res, "names") else {}
                rows = []
                for b in res.boxes:
                    cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                    xyxy = b.xyxy[0].tolist()
                    rows.append({
                        "label": names.get(cls_id, str(cls_id)),
                        "confidence": round(conf, 3),
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Объекты не найдены.")

            # Для наглядности — строка из вашего примера:
            st.code("/hdd/home/ABaglanova/indrive_hack/images.jpeg: 448x640 1 scratch, 1 shattered_glass, 9.2ms")
    except Exception as e:
        st.exception(e)

# -----------------------------
# Кнопка 2: VGG16 — загрязнения
# -----------------------------
if btn_dirty:
    st.subheader("🧼 Поиск загрязнений (VGG16)")
    try:
        # Источник берём тот же: загруженный файл или дефолтный путь
        source = make_yolo_source(uploaded, YOLO_TEST_IMG)
        if source is None:
            st.error("Нет изображения: загрузите файл или проверьте путь по умолчанию.")
        else:
            # Для VGG используем PIL.Image (если пришёл путь, откроем)
            img = source if isinstance(source, Image.Image) else open_image_any(source)

            model_vgg = build_vgg16_binary(VGG16_WEIGHTS)
            tfm = vgg_transform()
            with torch.no_grad():
                x = tfm(img).unsqueeze(0)
                logits = model_vgg(x)
                prob = torch.softmax(logits, dim=1)[0]
                # Допустим: индекс 0 — clean, индекс 1 — dirty (при необходимости поменяйте местами)
                p_clean = float(prob[0].item())
                p_dirty = float(prob[1].item())

            col_img, col_stats = st.columns([2, 1])
            with col_img:
                st.image(img, caption="Исходное изображение", use_column_width=True)
            with col_stats:
                st.metric("Вероятность 'загрязнено'", f"{p_dirty*100:.1f}%")
                st.progress(min(1.0, p_dirty))
                st.caption(f"clean: {p_clean:.3f} • dirty: {p_dirty:.3f}")

            verdict = "ЗАГРЯЗНЕНО" if p_dirty >= 0.5 else "ЧИСТО"
            st.success(f"Вердикт: {verdict}")
    except Exception as e:
        st.exception(e)

# -----------------------------
# Кнопка 3: Регистрация (заглушка)
# -----------------------------
if btn_reg:
    st.subheader("🪪 Пройти регистрацию")
    st.info("Страница регистрации в разработке. Здесь появится форма с полями и валидацией.")
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Имя")
        phone = st.text_input("Телефон")
        email = st.text_input("E-mail")
        agree = st.checkbox("Согласен на обработку персональных данных")
        submitted = st.form_submit_button("Отправить")
        if submitted:
            if not agree:
                st.error("Для продолжения необходимо согласие на обработку данных.")
            else:
                st.success("Заявка отправлена! (демо)")

st.markdown("---")
st.caption("© 2025 • Demo App: YOLOv8 + VGG16")
