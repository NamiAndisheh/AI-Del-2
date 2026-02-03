2. Installera dependencies:
```bash
pip install -r requirements.txt
pip install pypdf google-genai
```

3. Skapa en `.env` fil i `my_dir/` mappen:
```bash
cd my_dir
echo "API_KEY=din_google_gemini_api_nyckel" > .env
```

Kör applikationen

Lokalt (dator)

Kör följande kommando i terminalen:

```bash
python3 -m streamlit run "my_dir/Övningar/kunskapskontroll del2.py"
```

Applikationen kommer att öppnas i din webbläsare på `http://localhost:8501`

På mobilen (via ngrok)

1. Först, starta Streamlit-applikationen (se ovan)

2. I en ny terminal, kör ngrok:
```bash
ngrok http 8501
```

3. Öppna den publika länken som ngrok visar (t.ex. `https://xxxx-xxxx-xxxx.ngrok-free.dev`) i din mobilens webbläsare
