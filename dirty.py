from typing import Union, Dict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ====== НАСТРОЙКИ ======
DEFAULT_WEIGHTS = "/hdd/home/ABaglanova/indrive_hack/vgg16_dirty_clean_best.pth"
# По умолчанию считаем, что 0 = clean, 1 = dirty
DEFAULT_LABEL_MAP: Dict[int, str] = {0: "clean", 1: "dirty"}

_img_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _to_pil(img: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        # Если BGR (часто из cv2), переведём в RGB
        if img.ndim == 3 and img.shape[2] == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # эвристика: если пришло от cv2 — вероятно BGR, развернём каналы
            img = img[..., ::-1]
        return Image.fromarray(img).convert("RGB")
    raise TypeError("Unsupported image type. Pass path, PIL.Image, or np.ndarray.")


@lru_cache(maxsize=1)
def _load_vgg16_binary(weights_path: str, num_classes: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)

    state = torch.load(weights_path, map_location=device)
    # поддержка разных форматов сохранения
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # убрать префикс "module." если тренировали с DataParallel
    cleaned = {}
    for k, v in (state.items() if isinstance(state, dict) else []):
        cleaned[k[7:]] = v if k.startswith("module.") else v

    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(cleaned or state, strict=False)

    model.to(device).eval()
    return model, device


def vgg_has_confident_dirty(
    image: Union[str, Path, np.ndarray, Image.Image],
    threshold: float = 0.7,
    weights: str = DEFAULT_WEIGHTS,
    label_map: Dict[int, str] = DEFAULT_LABEL_MAP,
    positive_label: str = "dirty",
) -> bool:
    """
    True, если вероятность класса `positive_label` (по умолчанию 'dirty') >= threshold.
    """
    model, device = _load_vgg16_binary(weights)
    pil = _to_pil(image)
    x = _img_tf(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # индекс целевого класса по названию
    pos_idx = next(i for i, name in label_map.items() if name == positive_label)
    return float(probs[pos_idx]) >= threshold


# ===== Пример =====
# is_dirty = vgg_has_confident_dirty("images.jpeg", threshold=0.7)
# print(is_dirty)
