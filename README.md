# 🚕InDrive — Контроль качества
<p align="center">
  <img src="ChatGPT Image 13 сент. 2025 г., 23_43_00.png" width="600" alt="InDrive QC UI"/>
</p>
Мини-гайд, чтобы всё взлетело за 1 минуту ⚡️

## 1) 🧰 Установка
```bash
cd Indrive
python -m venv .venv
source .venv/bin/activate     # Win: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```
## 2) ⬇️ Веса (Google Drive)
```bash
https://drive.google.com/drive/folders/1XAqvNpCRRqtgSnXA6tNFSes8bUZ4Hfad
```
## 3) ▶️ Запуск Demo 
```bash
streamlit run my_models.py
```

## 4) ▶️ Запуск Demo with Open AI KEY 
Добавьте API в 19 строчку
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

```bash
streamlit run demo_with_openai_key.py
```

## 📁 Структура
```bash
Indrive/
├─ my_models.py
├─ damage.py
├─ dirty.py
├─ trained.pt - добавить веса
├─ vgg16_dirty_clean_best.pth - добавить веса
├─ demo_with_vlm.py - demo with vlm
├─ demo_with_openai_key.py - demo with open ai
├─ car_seg_training_code - код тренировки модели сегментации
├─ car_clean - код тренировки чистоты машины, датасет в гугл драйве
├─ failed_apps - демки которые работают, но не прошли в прод
└─ sample_images_for_tests/ - для проверки моделей 
```

