\documentclass[12pt,a4paper]{article}

% --- Unicode + RU + Emoji-friendly fonts ---
\usepackage{fontspec}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\setmainfont{Noto Sans}
\setsansfont{Noto Sans}
\setmonofont{JetBrains Mono}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible}
\usepackage[margin=1in]{geometry}

\begin{document}

\begin{center}
{\LARGE 🚕🟩 InDrive --- \textbf{Контроль качества}}\\[4pt]
{\large Мини-гайд, чтобы всё взлетело за 1 минуту ⚡️}
\end{center}

\vspace{0.8em}

\section*{1) 🧰 Установка}
\begin{lstlisting}
cd Indrive
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
\end{lstlisting}

\section*{2) ⬇️ Веса с Google Drive}
Сохраняем прямо в корень репозитория под ваши имена файлов (или пропустите, если уже лежат).
\begin{lstlisting}
pip install gdown
gdown --fuzzy "https://drive.google.com/uc?id=<DAMAGE_ID>" -O trained.pt
gdown --fuzzy "https://drive.google.com/uc?id=<DIRTY_ID>"  -O vgg16_dirty_clean_best.pth
\end{lstlisting}

\section*{3) ▶️ Запуск}
\begin{lstlisting}
streamlit run my_models.py
\end{lstlisting}
Откроется \texttt{http://localhost:8501} 🔗

\section*{📁 Что где лежит}
\begin{lstlisting}
Indrive/
├─ my_models.py   # вход для Streamlit
├─ damage.py      # YOLO (повреждения)
├─ dirty.py       # VGG (грязь)
├─ trained.pt     # веса YOLO
├─ vgg16_dirty_clean_best.pth  # веса VGG
└─ sample_images_for_tests/    # тестовые картинки
\end{lstlisting}

\section*{💡 Подсказки}
\begin{itemize}[leftmargin=1.2em]
  \item ❗ Если видите ошибки CUDA/cuDNN --- используйте CPU-сборку PyTorch или поставьте колёса под вашу версию CUDA.
  \item 🧩 Хотите хранить веса вне корня? Укажите пути переменными окружения:
\end{itemize}

\begin{lstlisting}
# Linux/macOS:
export DAMAGE_WEIGHTS=models/yolo/damage_yolo.pt
export DIRTY_WEIGHTS=models/vgg/dirty_vgg.pt
