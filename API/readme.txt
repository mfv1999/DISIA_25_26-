Para arrancar

pip install -r requirements.txt
python run.py

== PARA PROBAR API ==

python -m venv .venv //PARA NO CONTAMINAR TU ENTORNO GLOBAL DE PYTHON
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')" //PARA QUE CUANDO PRUEBES LA EXTRACCION DE CARACTERISITICAS SAQUE TAMBIEN LAS DE VADER, SI NO LAS DEJABA A NULL
Y YA A PARTIR DE AQUI PUEDES:
- USAR EL SCRIPT DE GONZALO: python run.py
- O EJECUTAR MANUALMENTE EL SERVIDOR: uvicorn app.main:app --host 0.0.0.0 --port 8000 (O uvicorn app.main:app --reload, en lo práctico es lo mismo)