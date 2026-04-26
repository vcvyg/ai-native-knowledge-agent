$ErrorActionPreference = "Stop"
Set-Location "D:\ai-native-knowledge-agent"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8015 --reload

