# 🚕InDrive — Контроль качества

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
├─ trained.pt
├─ vgg16_dirty_clean_best.pth
└─ sample_images_for_tests/
```

