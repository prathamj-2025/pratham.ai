"""
Pratham.ai — a personal assistant chatbot.

A visitor (recruiter, hiring manager, curious person) can ask questions about
Pratham. The bot answers conversationally, grounded ONLY in profile.txt.
Uses Google Gemini 2.5 Flash-Lite (cheapest tier, free for low traffic).

To run locally:   streamlit run app.py

NOTE: All the facts about Pratham live in `profile.txt` — edit that file to
update what the bot knows. You don't need to touch this file to change content.
"""

from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────
# CONFIG  — easy knobs to tweak
# ──────────────────────────────────────────────────────────────────────────
ASSISTANT_NAME = "Pratham.ai"
MODEL = "gemini-2.5-flash-lite"   # ← change to "gemini-2.5-flash" for nicer answers
MESSAGE_LIMIT = 10                # ← max user messages per session; raise anytime
PROFILE_FILE = "profile.txt"      # ← the knowledge base lives here
GREETING = (
    "Hi! I'm Pratham.ai, Pratham's personal assistant. "
    "I can tell you about his experience, projects, skills, or what kind of "
    "roles he's looking for. What would you like to know?"
)

# Suggestion chips shown before the user types anything
OPENING_CHIPS = [
    "What's Pratham's background?",
    "What did he do at Quantiphi?",
    "What roles is he looking for?",
    "What's his tech stack?",
]


# ──────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  — loaded from profile.txt
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_profile():
    path = Path(__file__).parent / PROFILE_FILE
    if not path.exists():
        st.error(f"Could not find {PROFILE_FILE}. Make sure it's in the same "
                 "folder as app.py.")
        st.stop()
    return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  — defines the bot's persona and rules
# ──────────────────────────────────────────────────────────────────────────
def build_system_prompt():
    knowledge_base = load_profile()
    return f"""
You are {ASSISTANT_NAME}, Pratham Jain's personal assistant. Visitors — often
recruiters or hiring managers — chat with you to learn about Pratham.

HOW TO SPEAK:
- Always refer to Pratham in the THIRD PERSON ("Pratham did...", "he built...").
  You are his assistant, not Pratham himself. Never say "I" to mean Pratham.
- Be warm, natural, and conversational — like a knowledgeable friend describing
  him. NEVER paste resume bullet points verbatim. Rephrase facts into smooth,
  human sentences.
- Keep answers MEDIUM length: a short, substantive paragraph or two. Enough to be
  useful, not a wall of text. The user can always ask for more or tap a follow-up.

GROUNDING RULES:
- Answer ONLY using the profile below. Do not invent facts, dates, numbers,
  employers, or skills that aren't there.
- If a specific detail isn't in the profile (e.g. whether a company is a family
  business, salary expectations, personal life, an opinion he hasn't stated),
  simply say you don't have that particular detail about Pratham — in a natural,
  friendly way. Do NOT mention "knowledge base", "profile", "my data", or anything
  about how you're built. Just say something like "I don't have that detail about
  Pratham" and offer a helpful related direction. Never make something up.
- If asked something completely unrelated to Pratham (coding help, the weather,
  general trivia), kindly explain you're here specifically to talk about Pratham,
  and offer a relevant direction instead.

FOLLOW-UP QUESTIONS:
- At the very end of EVERY reply, suggest 2–3 natural follow-up questions the
  visitor might want to ask next, based on what was just discussed.
- Format them EXACTLY like this, each on its own line, nothing after them:
  [SUGGESTIONS]
  First follow-up question?
  Second follow-up question?
  Third follow-up question?
- Phrase suggestions in third person too ("What did Pratham build at Quantiphi?"),
  never "What did you build?".

PROFILE:
{knowledge_base}
"""


# ──────────────────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error(
            "No API key found. Add GEMINI_API_KEY to your Streamlit secrets "
            "(see README)."
        )
        st.stop()
    return genai.Client(api_key=api_key)


def parse_reply(text):
    """Split the model output into the answer and a list of follow-up chips."""
    if "[SUGGESTIONS]" in text:
        answer, _, raw = text.partition("[SUGGESTIONS]")
        chips = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return answer.strip(), chips[:3]
    return text.strip(), []


def get_response(history):
    """Send conversation history to Gemini and return (answer, chips)."""
    client = get_client()
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(),
                temperature=0.7,
                max_output_tokens=600,
            ),
        )
        return parse_reply(resp.text)
    except Exception:
        return (
            "Sorry, I'm having trouble connecting right now. Please try again "
            "in a moment.",
            [],
        )


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=ASSISTANT_NAME, page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stChatMessage { font-size: 1rem; }
    .block-container { padding-top: 2rem; max-width: 720px; }
    div.stButton > button {
        text-align: left; justify-content: flex-start;
        border-radius: 10px; padding: 0.6rem 0.9rem; width: 100%;
        border: 1px solid rgba(128,128,128,0.25); background: transparent;
        font-weight: 400;
    }
    div.stButton > button:hover { border-color: rgba(128,128,128,0.6); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title(f"💬 {ASSISTANT_NAME}")
st.caption("Pratham Jain's personal assistant · ask me anything about his work")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chips" not in st.session_state:
    st.session_state.chips = OPENING_CHIPS
if "pending" not in st.session_state:
    st.session_state.pending = None

# Reset button
col1, col2 = st.columns([5, 1])
with col2:
    if st.button("Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chips = OPENING_CHIPS
        st.session_state.pending = None
        st.rerun()

# Greeting (always shown first)
with st.chat_message("assistant", avatar="💬"):
    st.markdown(GREETING)

# Render conversation so far
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "💬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Count how many messages the visitor has sent
user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
limit_reached = user_turns >= MESSAGE_LIMIT


def submit(text):
    st.session_state.pending = text


# Suggestion chips (clickable)
if st.session_state.chips and not limit_reached and st.session_state.pending is None:
    label = "Continue exploring:" if st.session_state.messages else "Try asking:"
    st.markdown(f"**{label}**")
    for i, chip in enumerate(st.session_state.chips):
        st.button(f"→ {chip}", key=f"chip_{len(st.session_state.messages)}_{i}",
                  on_click=submit, args=(chip,))

# Chat input
if limit_reached:
    st.chat_input("Message limit reached for this session — tap Reset to start over.",
                  disabled=True)
    st.info("You've reached the message limit for this session. Tap **Reset** above "
            "to start a new conversation.")
else:
    typed = st.chat_input("Ask me something about Pratham...")
    if typed:
        submit(typed)

# Process a pending message (from a chip click or typed input)
if st.session_state.pending is not None:
    user_text = st.session_state.pending
    st.session_state.pending = None
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Thinking..."):
            answer, chips = get_response(st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chips = chips
    st.rerun()

# Footer
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem; margin-top:2rem;'>"
    f"{ASSISTANT_NAME} can make mistakes. Please verify important details.</p>",
    unsafe_allow_html=True,
)
