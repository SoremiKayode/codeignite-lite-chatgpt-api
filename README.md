# 1) Start Ollama daemon (if not already running)
ollama serve

# 2) Ensure the model is available (adjust tag if different on your machine)
ollama pull qwen2.5:3b-instruct-q4_K_M

# 3) Start the FastAPI app
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
