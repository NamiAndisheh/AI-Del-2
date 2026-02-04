
1. Skapa en `.env`:
"API_KEY=din_google_gemini_api_nyckel" > .env

2. Kör följande kommando i terminalen:

python3 -m streamlit run "my_dir/Övningar/kunskapskontroll del2.py"

På mobilen (via ngrok)

3. I en ny terminal, kör ngrok:
   
ngrok http 8501

4. Öppna den publika länken som ngrok visar (t.ex. `https://xxxx-xxxx-xxxx.ngrok-free.dev`) i din mobilens webbläsare
