FROM python:3.10-slim

# 🔧 Встановлюємо системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential jq gcc libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 🔧 Створюємо робочу директорію
WORKDIR /workspace

# 🔧 Оновлюємо pip
RUN pip install --upgrade pip setuptools wheel

# 📦 Встановлюємо ядро PromptBench + JupyterLab
RUN pip install --no-cache-dir \
    jupyterlab ipykernel \
    promptbench \
    transformers \
    openai==0.27.8 \
    httpx==0.24.1 \
    anyio==3.7.1

# 🧠 Додатково: бібліотеки для prompt-level атак
RUN pip install --no-cache-dir \
    textattack \
    tensorflow \
    tensorflow_hub

# 🧪 (Опційно) встановити ядро в Jupyter
RUN python3 -m ipykernel install --user --name=promptbench --display-name "promptbench"

# 🗂️ Створюємо папку для збереження прикладів
RUN mkdir -p /workspace/examples

# 📥 Копіюємо скрипти (опційно)
# COPY run_security_eval.py /workspace/
# COPY run_prompt_attack.py /workspace/

# 🔧 Відкриваємо порт для Jupyter
EXPOSE 8888

# 🚀 Запускаємо JupyterLab при запуску контейнера
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

