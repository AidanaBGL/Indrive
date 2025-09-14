from typing import Union, Optional, List
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


@lru_cache(maxsize=1)
def _load_model(weights_path: str) -> YOLO:
    """Ленивая загрузка YOLO, кэш по пути к весам."""
    return YOLO(weights_path)


def yolo_has_confident_detection(
    image: Union[str, Path, np.ndarray, Image.Image],
    threshold: float = 0.7,
    weights: str = "trained.pt",
    classes: Optional[List[int]] = None,   # можно указать фильтр по классам, например [0, 1]
) -> bool:
    """
    Возвращает True, если есть хотя бы одна детекция с conf >= threshold.
    :param image: путь к файлу, PIL.Image или np.ndarray (HWC, BGR/RGB — как в ultralytics)
    :param threshold: порог confidence
    :param weights: путь к .pt весам
    :param classes: опционально — фильтр по ID классов YOLO
    """
    model = _load_model(weights)
    results = model(image, classes=classes, verbose=False)

    if not results:
        return False

    r = results[0]
    boxes = getattr(r, "boxes", None)
    # нет боксов
    if boxes is None or boxes.conf is None or len(boxes) == 0:
        return False

    # max confidence по всем детекциям
    max_conf = float(boxes.conf.max().item())
    return max_conf >= threshold


# Пример использования:
# ok = yolo_has_confident_detection("/hdd/home/ABaglanova/indrive_hack/images.jpeg", threshold=0.7)
# print(ok)
