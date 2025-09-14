# 🚕🟩 InDrive — Контроль качества

Мини-гайд, чтобы всё взлетело за 1 минуту ⚡️

## 1) 🧰 Установка
```bash
cd Indrive
python -m venv .venv
source .venv/bin/activate     # Win: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

pip install gdown
gdown --fuzzy "https://drive.google.com/uc?id=<DAMAGE_ID>" -O trained.pt
gdown --fuzzy "https://drive.google.com/uc?id=<DIRTY_ID>"  -O vgg16_dirty_clean_best.pth

streamlit run my_models.py

Indrive/
├─ my_models.py
├─ damage.py
├─ dirty.py
├─ trained.pt
├─ vgg16_dirty_clean_best.pth
└─ sample_images_for_tests/
