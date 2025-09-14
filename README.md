🚕🟩 InDrive — Контроль качества

Мини-гайд, чтобы всё взлетело за 1 минуту ⚡️

1) 🧰 Установка
cd Indrive
python -m venv .venv
source .venv/bin/activate     # Win: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

2) ⬇️ Веса с Google Drive

Сохраняем прямо в корень, под ваши имена файлов.

pip install gdown
gdown --fuzzy "https://drive.google.com/uc?id=<DAMAGE_ID>" -O trained.pt
gdown --fuzzy "https://drive.google.com/uc?id=<DIRTY_ID>"  -O vgg16_dirty_clean_best.pth


(если уже лежат — этот шаг пропускаем ✅)

3) ▶️ Запуск
streamlit run my_models.py


Откроется http://localhost:8501 🔗

📁 Что где лежит
Indrive/
├─ my_models.py   # вход для Streamlit
├─ damage.py      # YOLO (повреждения)
├─ dirty.py       # VGG (грязь)
├─ trained.pt     # веса YOLO
├─ vgg16_dirty_clean_best.pth  # веса VGG
└─ sample_images_for_tests/    # тестовые картинки

💡 Подсказки

Ошибка CUDA/cuDNN? Поставьте CPU-сборку PyTorch или подходящую под вашу CUDA.

Хотите хранить веса отдельно?

export DAMAGE_WEIGHTS=models/yolo/damage_yolo.pt
export DIRTY_WEIGHTS=models/vgg/dirty_vgg.pth


Ура, готово! 🚀