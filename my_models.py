import io
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import streamlit as st
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Taxi Driver QC: Damage / Dirt / Registration", layout="wide")

# ================== IMPORT YOUR MODELS ==================
# They must exist in your project next to this file as damage.py and dirty.py
# Each should expose the following call signatures:
#   yolo_has_confident_detection(image_path: str | Path, threshold: float) -> bool | Dict[str, Any]
#   vgg_has_confident_dirty(image_path: str | Path, threshold: float) -> bool | Dict[str, Any]
try:
    from damage import yolo_has_confident_detection  # noqa
except Exception as e:
    yolo_has_confident_detection = None
    _damage_import_error = str(e)

try:
    from dirty import vgg_has_confident_dirty  # noqa
except Exception as e:
    vgg_has_confident_dirty = None
    _dirty_import_error = str(e)

# ================== CLASSIFICATION RULES (BRAND/MODEL/YEAR) ==================
brand_classes = {
    "Toyota": "comfort",
    "Hyundai": "econom",
    "Kia": "econom",
    "BMW": "business",
    "Mercedes-Benz": "business+",
    "Audi": "business",
    "Lexus": "business+",
    "Bentley": "prime",
    "Rolls-Royce": "prime",
}

model_classes = {
    "Camry": "+",
    "Corolla": "=",
    "Elantra": "=",
    "S-Class": "++",
    "3 Series": "=",
    "7 Series": "+",
    "Phantom": "prime",
}

year_thresholds = [
    (2010, "comfort+"),
    (2018, "business+"),
    (2025, "prime"),
]

class_order = ["econom", "comfort", "comfort+", "business", "business+", "prime"]


def adjust_class(base: str, adjustment: str) -> str:
    idx = class_order.index(base)
    if adjustment == "=":
        return base
    elif adjustment == "+" and idx < len(class_order) - 1:
        return class_order[idx + 1]
    elif adjustment == "++" and idx < len(class_order) - 2:
        return class_order[idx + 2]
    elif adjustment in class_order:
        return adjustment
    return base


def cap_by_year(year: int, current_class: str) -> str:
    for limit, cap in year_thresholds:
        if year <= limit:
            return min(current_class, cap, key=lambda x: class_order.index(x))
    return current_class


def classify_car(brand: str, model: str, year: int) -> str:
    base_class = brand_classes.get(brand, "econom")
    adjustment = model_classes.get(model, "=")
    adjusted_class = adjust_class(base_class, adjustment)
    final_class = cap_by_year(year, adjusted_class)
    return final_class

# ================== HELPERS ==================
SIDE_ORDER = ["front", "left", "back", "right"]
SIDE_LABELS = {"front": "Front", "left": "Left", "back": "Back", "right": "Right"}


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def save_uploaded_image(uploaded_file, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix or ".jpg"
    out = workdir / f"upload{ext}"
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out


# ================== SESSION STATE ==================
if "registration" not in st.session_state:
    st.session_state.registration = {
        "first_name": "",
        "last_name": "",
        "brand": None,
        "model": None,
        "year": 2015,
        "class": None,
        "doc_photo": None,  # PIL bytes
        "doc_verified": None,  # tuple (matched, distance, error, num_faces)
    }

if "sides" not in st.session_state:
    st.session_state.sides = {k: {"shot": None, "damage": None, "dirty": None} for k in SIDE_ORDER}

if "damage" not in st.session_state:
    st.session_state.damage = {"checked": False, "per_side": {}, "threshold": 0.7}

if "dirty" not in st.session_state:
    st.session_state.dirty = {"checked": False, "per_side": {}, "threshold": 0.7}


# ================== SIDEBAR (THREE BUTTONS) ==================
st.sidebar.header("Действия")
page = st.sidebar.radio(
    "Выберите задачу:",
    (
        "Найти повреждения",
        "Найти загрязнения",
        "Пройти регистрацию",
    ),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Models: YOLO (damage) & VGG16 (dirty)")

# ================== PAGES ==================

# 1) DAMAGE PAGE
if page == "Найти повреждения":
    st.title("Проверка повреждений")
    st.write("Загрузите изображение. Будет запущена модель повреждений.")

    threshold = st.slider("Порог повреждений", 0.1, 0.99, st.session_state.damage["threshold"], 0.01)
    st.session_state.damage["threshold"] = threshold

    col_up, col_out = st.columns([1, 2])
    with col_up:
        f = st.file_uploader("Изображение машины", type=["jpg", "jpeg", "png"])
        run_btn = st.button("Проверить повреждения")
    with col_out:
        if run_btn and f is not None:
            img_path = save_uploaded_image(f, Path(".cache/damage"))
            if yolo_has_confident_detection is None:
                st.error(f"Модель повреждений не импортирована: {_damage_import_error}")
            else:
                try:
                    result = yolo_has_confident_detection(str(img_path), threshold=threshold)
                    st.success(f"Damage result: {result}")
                except Exception as e:
                    st.exception(e)
        elif run_btn and f is None:
            st.warning("Пожалуйста, загрузите изображение.")

# 2) DIRTY PAGE
elif page == "Найти загрязнения":
    st.title("Проверка загрязнений")
    st.write("Загрузите изображение. Будет запущена модель чистоты (грязно/чисто).")

    threshold = st.slider("Порог загрязнений", 0.1, 0.99, st.session_state.dirty["threshold"], 0.01)
    st.session_state.dirty["threshold"] = threshold

    col_up, col_out = st.columns([1, 2])
    with col_up:
        f = st.file_uploader("Изображение машины", type=["jpg", "jpeg", "png"])
        run_btn = st.button("Проверить загрязнение")
    with col_out:
        if run_btn and f is not None:
            img_path = save_uploaded_image(f, Path(".cache/dirty"))
            if vgg_has_confident_dirty is None:
                st.error(f"Модель чистоты не импортирована: {_dirty_import_error}")
            else:
                try:
                    result = vgg_has_confident_dirty(str(img_path), threshold=threshold)
                    st.success(f"Dirty result: {result}")
                except Exception as e:
                    st.exception(e)
        elif run_btn and f is None:
            st.warning("Пожалуйста, загрузите изображение.")

# 3) REGISTRATION PAGE
elif page == "Пройти регистрацию":
    st.title("Регистрация водителя")

    with st.form("reg_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.registration["first_name"] = st.text_input("Имя", st.session_state.registration["first_name"])
            st.session_state.registration["last_name"] = st.text_input("Фамилия", st.session_state.registration["last_name"])
            brand = st.selectbox("Бренд", [None] + list(brand_classes.keys()), index=(0 if st.session_state.registration["brand"] is None else 1 + list(brand_classes.keys()).index(st.session_state.registration["brand"])) )
            model = st.selectbox("Модель", [None] + list(model_classes.keys()), index=(0 if st.session_state.registration["model"] is None else 1 + list(model_classes.keys()).index(st.session_state.registration["model"])) )
            year = st.number_input("Год", value=int(st.session_state.registration["year"]), min_value=1990, max_value=2025, step=1)
        with c2:
            st.markdown("**Фото с документом** (селфи + фото в одном кадре)")
            doc_file = st.file_uploader("Загрузите фото документа и лица", type=["jpg", "jpeg", "png"], key="docup")
            submit = st.form_submit_button("Сохранить и рассчитать класс")

        if submit:
            st.session_state.registration["brand"] = brand
            st.session_state.registration["model"] = model
            st.session_state.registration["year"] = int(year)

            if brand and model and year:
                assigned = classify_car(brand, model, int(year))
                st.session_state.registration["class"] = assigned
                st.success(f"Класс авто: {assigned}")
            else:
                st.warning("Заполните бренд, модель и год.")

            if doc_file is not None:
                # Keep raw bytes for next step (face verification outside of this minimal demo)
                st.session_state.registration["doc_photo"] = doc_file.getvalue()
                st.image(doc_file, caption="Загруженное фото документа+лица", use_column_width=True)

    st.markdown("---")
    st.subheader("Шаг 2: Ввод/проверка лиц (опционально)")
    st.caption("Подключите вашу функцию verify_two_faces_in_one(image) и добавьте кнопку проверки. В этом шаблоне мы только сохраняем фото.")

    st.markdown("---")
    st.subheader("Шаг 3: Фото по сторонам")
    st.caption("Сделайте по одной фотографии на каждую сторону: Front, Left, Back, Right")

    # Uploaders per side
    cols = st.columns(4)
    up_keys = ["up_front", "up_left", "up_back", "up_right"]
    for i, side in enumerate(SIDE_ORDER):
        with cols[i]:
            st.markdown(f"**{SIDE_LABELS[side]}**")
            uf = st.file_uploader("Загрузить", type=["jpg", "jpeg", "png"], key=up_keys[i])
            if uf is not None:
                path = save_uploaded_image(uf, Path(f".cache/side/{side}"))
                st.session_state.sides[side]["shot"] = str(path)
                st.image(uf, caption=f"{SIDE_LABELS[side]}")

    st.markdown("---")
    st.subheader("Проверка сторон: Damage + Dirty")

    c1, c2 = st.columns(2)
    with c1:
        dmg_th = st.slider("Порог Damage", 0.1, 0.99, st.session_state.damage["threshold"], 0.01, key="dmg_chk")
    with c2:
        dirt_th = st.slider("Порог Dirty", 0.1, 0.99, st.session_state.dirty["threshold"], 0.01, key="dirt_chk")

    run_all = st.button("Проверить все стороны")

    if run_all:
        per_side: Dict[str, Dict[str, Any]] = {}
        any_damage = False
        any_dirty = False
        for side in SIDE_ORDER:
            shot = st.session_state.sides[side]["shot"]
            if not shot:
                per_side[side] = {"is_damaged": None, "is_dirty": None, "error": "no image"}
                continue
            entry: Dict[str, Any] = {}
            # Damage
            if yolo_has_confident_detection is None:
                entry["is_damaged"] = None
                entry["damage_error"] = _damage_import_error
            else:
                try:
                    dres = yolo_has_confident_detection(shot, threshold=dmg_th)
                    entry["is_damaged"] = bool(dres)
                    any_damage = any_damage or bool(dres)
                except Exception as e:
                    entry["is_damaged"] = None
                    entry["damage_error"] = str(e)
            # Dirty
            if vgg_has_confident_dirty is None:
                entry["is_dirty"] = None
                entry["dirty_error"] = _dirty_import_error
            else:
                try:
                    vres = vgg_has_confident_dirty(shot, threshold=dirt_th)
                    entry["is_dirty"] = bool(vres)
                    any_dirty = any_dirty or bool(vres)
                except Exception as e:
                    entry["is_dirty"] = None
                    entry["dirty_error"] = str(e)

            per_side[side] = entry

        st.session_state.damage["per_side"] = per_side
        st.session_state.damage["checked"] = True
        st.session_state.damage["threshold"] = dmg_th
        st.session_state.dirty["per_side"] = per_side
        st.session_state.dirty["checked"] = True
        st.session_state.dirty["threshold"] = dirt_th

    # Render results if exist
    per_side = st.session_state.damage.get("per_side", {})
    if per_side:
        st.write("**Результаты по сторонам:**")
        for k in SIDE_ORDER:
            r = per_side.get(k, {})
            line = f"**{SIDE_LABELS[k]}** — "
            dmg = r.get("is_damaged")
            drt = r.get("is_dirty")
            if dmg is None and drt is None:
                line += "_нет изображения или модели недоступны_"
            else:
                if dmg is None:
                    line += "Damage: _n/a_ "
                else:
                    line += f"Damage: {'🔴' if dmg else '🟢'}  "
                if drt is None:
                    line += "Dirty: _n/a_"
                else:
                    line += f"Dirty: {'🔴' if drt else '🟢'}"
            st.write(line)

        st.markdown("---")
        any_damage = any(bool(per_side.get(k, {}).get("is_damaged")) for k in SIDE_ORDER)
        any_dirty = any(bool(per_side.get(k, {}).get("is_dirty")) for k in SIDE_ORDER)
        if any_damage or any_dirty:
            st.error("Итог: **Не подходит** для такси (обнаружены повреждения и/или сильные загрязнения).")
        else:
            st.success("Итог: **Подходит** для такси (повреждения/загрязнения не обнаружены выше порога).")

    st.caption("Вы можете повторно загрузить фотографии и нажать 'Проверить все стороны'.")
