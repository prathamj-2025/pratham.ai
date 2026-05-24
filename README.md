# Pratham.ai — personal assistant chatbot

A simple, conversational chatbot that answers questions about Pratham Jain.
Built with Streamlit + Google Gemini 2.5 Flash-Lite.

## What's here
- `app.py` — the whole app (chat UI + bot logic + your knowledge base)
- `requirements.txt` — dependencies

---

## 1. Get a free Gemini API key
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account → "Create API key".
3. Copy the key. (Flash-Lite has a generous free tier — low traffic likely costs $0.)

---

## 2. Run it locally (to test)
```bash
pip install -r requirements.txt
```
Create a file at `.streamlit/secrets.toml` with:
```toml
GEMINI_API_KEY = "your-key-here"
```
Then run:
```bash
streamlit run app.py
```
It opens at http://localhost:8501

---

## 3. Deploy it free (public URL)
1. Push this folder to a **GitHub** repo.
   - Do NOT commit your key. Make sure `.streamlit/secrets.toml` is in `.gitignore`.
2. Go to https://share.streamlit.io → "New app" → connect your repo → pick `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy. You'll get a public URL like `https://pratham-ai.streamlit.app` to share.

---

## Easy things to change (top of `app.py`)
- `ASSISTANT_NAME` — the bot's name.
- `MODEL` — `"gemini-2.5-flash-lite"` (cheapest) or `"gemini-2.5-flash"` (nicer answers).
- `MESSAGE_LIMIT` — messages per session. Change `10` to any number.
- `GREETING` — the opening message.
- `OPENING_CHIPS` — the suggestion buttons shown before the user types.
- `KNOWLEDGE_BASE` — everything the bot knows about Pratham. **Edit this freely
  to add more info later** — just plain text, no rebuild needed.

## Note
`.gitignore` should include:
```
.streamlit/secrets.toml
```
