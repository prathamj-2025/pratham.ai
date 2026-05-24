"""
Pratham.ai — a personal assistant chatbot.

A visitor (recruiter, hiring manager, curious person) can ask questions about
Pratham. The bot answers conversationally, grounded ONLY in profile.txt.
Uses Google Gemini 2.5 Flash-Lite (cheapest tier, free for low traffic).

To run locally:   streamlit run app.py

NOTE: All the facts about Pratham live in `profile.txt` — edit that file to
update what the bot knows. You don't need to touch this file to change content.

ERROR HANDLING:
- Quota / API-key problems  -> apologize + capture the visitor's email to a
  Google Sheet so Pratham can notify them when it's back.
- Rate limit (too busy)     -> ask them to wait a few seconds and retry.
- Network / timeout blip    -> ask them to refresh.
- Empty / odd response      -> ask them to rephrase.
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
    "Hi there! I'm Pratham.ai, Pratham's personal AI assistant. "
    "I'd be happy to tell you about his education, work experience, or any "
    "other aspect of his professional journey. What would you like to know?"
)

# Suggestion chips shown before the user types anything
OPENING_CHIPS = [
    "What's Pratham's background?",
    "What did he do at Quantiphi?",
    "What roles is he looking for?",
    "What's his tech stack?",
]

# Error category messages shown in chat
MSG_RATE_LIMIT = (
    "I'm getting a lot of questions right now! Please give it a few seconds "
    "and try again."
)
MSG_NETWORK = (
    "Looks like a brief connection hiccup. Please refresh the page or try "
    "again in a moment."
)
MSG_REPHRASE = (
    "Sorry, I didn't quite catch that. Could you rephrase your question?"
)
MSG_UNAVAILABLE = (
    "I'm so sorry for the inconvenience — Pratham.ai is temporarily "
    "unavailable right now. If you'd like, leave your email below and I'll "
    "notify you the moment it's back, so you can continue learning about "
    "Pratham."
)


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
# SYSTEM PROMPT
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


def classify_error(err):
    """Map an exception to one of: 'unavailable', 'rate_limit', 'network'."""
    text = str(err).lower()
    # Quota exhausted or billing/key issues -> treat as a real outage.
    if any(k in text for k in
           ["quota", "exhausted", "billing", "permission", "api key",
            "api_key", "invalid", "unauthenticated", "credential", "403", "401"]):
        return "unavailable"
    # Too many requests right now -> temporary, ask to wait.
    if any(k in text for k in ["rate", "429", "resource_exhausted", "overloaded"]):
        return "rate_limit"
    # Everything else -> treat as a transient network problem.
    return "network"


def is_valid_email(email):
    """Basic sanity check: has text, one @, and a dot after it."""
    email = (email or "").strip()
    if email.count("@") != 1:
        return False
    local, _, domain = email.partition("@")
    return bool(local) and "." in domain and not domain.endswith(".")


def parse_reply(text):
    """Split the model output into the answer and a list of follow-up chips."""
    if "[SUGGESTIONS]" not in text:
        return text.strip(), []
    answer, _, raw = text.partition("[SUGGESTIONS]")
    raw = raw.strip()
    # First try splitting by line. If the model crammed them onto one line,
    # fall back to splitting on question marks.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 1 and "?" in raw:
        lines = [q.strip() + "?" for q in raw.split("?") if q.strip()]
    # Clean any leading bullet/number markers the model may add
    chips = []
    for ln in lines:
        ln = ln.lstrip("-*•0123456789. ").strip()
        if ln:
            chips.append(ln)
    return answer.strip(), chips[:3]


def get_response(history):
    """
    Send conversation history to Gemini.
    Returns (answer, chips, status) where status is one of:
    'ok', 'unavailable', 'rate_limit', 'network', 'rephrase'.
    """
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
        text = (resp.text or "").strip()
        if not text:
            return MSG_REPHRASE, [], "rephrase"
        answer, chips = parse_reply(text)
        return answer, chips, "ok"
    except Exception as e:
        print("DEBUG ERROR:", repr(e))   # TEMP: remove after debugging
        kind = classify_error(e)
        if kind == "unavailable":
            return MSG_UNAVAILABLE, [], "unavailable"
        if kind == "rate_limit":
            return MSG_RATE_LIMIT, [], "rate_limit"
        return MSG_NETWORK, [], "network"


# ──────────────────────────────────────────────────────────────────────────
# EMAIL CAPTURE  — saves a waitlist email to a Google Sheet
# ──────────────────────────────────────────────────────────────────────────
def save_email(email):
    """
    Append an email to the configured Google Sheet.
    Returns True on success, False otherwise.
    Requires a [gcp_service_account] block and WAITLIST_SHEET_ID in secrets (see README).
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        from datetime import datetime, timezone

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        gc = gspread.authorize(creds)
        sheet = gc.open_by_key(st.secrets["WAITLIST_SHEET_ID"]).sheet1
        sheet.append_row([email, datetime.now(timezone.utc).isoformat()])
        return True
    except Exception:
        return False


def save_feedback(rating, comment, email):
    """
    Append feedback to the FEEDBACK Google Sheet (a separate sheet from the
    waitlist). Returns True on success, False otherwise.
    Requires FEEDBACK_SHEET_ID in secrets (see README).
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        from datetime import datetime, timezone

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        gc = gspread.authorize(creds)
        sheet = gc.open_by_key(st.secrets["FEEDBACK_SHEET_ID"]).sheet1
        sheet.append_row([
            rating, comment, email,
            datetime.now(timezone.utc).isoformat(),
        ])
        return True
    except Exception:
        return False


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

    /* Fixed disclaimer below the chat input — aligns with main content */
    .disclaimer-fixed {
        position: fixed;
        bottom: 6px;
        left: 0;
        right: 0;
        margin-left: auto;
        margin-right: auto;
        width: calc(100% - 250px); /* accounts for sidebar width */
        margin-left: 250px;        /* offset by sidebar */
        text-align: center;
        color: gray;
        font-size: 0.78rem;
        padding: 4px 0;
        background: white;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fixed disclaimer pinned below the chat input
st.markdown(
    f"<div class='disclaimer-fixed'>{ASSISTANT_NAME} can make mistakes. "
    "Please verify important details.</div>",
    unsafe_allow_html=True,
)

# Header
st.title(f"💬 Chat with {ASSISTANT_NAME}")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chips" not in st.session_state:
    st.session_state.chips = OPENING_CHIPS
if "pending" not in st.session_state:
    st.session_state.pending = None
if "show_email_capture" not in st.session_state:
    st.session_state.show_email_capture = False
if "email_saved" not in st.session_state:
    st.session_state.email_saved = False
if "feedback_open" not in st.session_state:
    st.session_state.feedback_open = False
if "feedback_saved" not in st.session_state:
    st.session_state.feedback_saved = False
if "feedback_just_saved" not in st.session_state:
    st.session_state.feedback_just_saved = False

# ── Sidebar: controls (always visible, don't scroll away with the chat) ──
with st.sidebar:
    st.markdown(f"### {ASSISTANT_NAME}")
    st.caption("Controls")
    if st.button("Feedback", use_container_width=True):
        st.session_state.feedback_open = True
        st.rerun()
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chips = OPENING_CHIPS
        st.session_state.pending = None
        st.session_state.show_email_capture = False
        st.session_state.email_saved = False
        st.session_state.feedback_open = False
        st.session_state.feedback_saved = False
        st.session_state.feedback_just_saved = False
        st.rerun()

    # Feedback form — opens here in the sidebar when "Feedback" is clicked
    if st.session_state.feedback_open and not st.session_state.feedback_saved:
        st.divider()
        st.markdown("**How was your experience?**")
        with st.form("feedback_form", clear_on_submit=True):
            rating = st.radio("Rating", ["👍 Helpful", "👎 Not helpful"],
                              label_visibility="collapsed")
            comment = st.text_area("Any comments? (optional)",
                                   placeholder="What worked well, or what "
                                   "could be better?")
            fb_email = st.text_input("Your email (optional)",
                                     placeholder="you@example.com")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                sent = st.form_submit_button("Submit", use_container_width=True)
            with col_b:
                cancelled = st.form_submit_button("Cancel",
                                                  use_container_width=True)
            if sent:
                if fb_email.strip() and not is_valid_email(fb_email):
                    st.warning("That email doesn't look right. Please fix it "
                               "or leave it blank.")
                else:
                    if save_feedback(rating, comment, fb_email.strip()):
                        st.session_state.feedback_saved = True
                        st.session_state.feedback_open = False
                        st.session_state.feedback_just_saved = True
                        st.rerun()
                    else:
                        st.error("Couldn't submit just now — please try again "
                                 "later.")
            elif cancelled:
                st.session_state.feedback_open = False
                st.rerun()

# Greeting (always shown first)
with st.chat_message("assistant", avatar="💬"):
    st.markdown(GREETING)

# Render conversation so far
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "💬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Email capture box (shown when the bot is unavailable) ──
if st.session_state.show_email_capture and not st.session_state.email_saved:
    with st.form("email_capture", clear_on_submit=True):
        email = st.text_input("Your email", placeholder="you@example.com")
        submitted = st.form_submit_button("Notify me when it's back")
        if submitted:
            if is_valid_email(email):
                if save_email(email):
                    st.session_state.email_saved = True
                    st.rerun()
                else:
                    st.error("Couldn't save that just now — please try again "
                             "later.")
            else:
                st.warning("Please enter a valid email address.")
elif st.session_state.email_saved:
    st.success("Thanks! Your email is saved — you'll be notified when "
               f"{ASSISTANT_NAME} is back.")

# Count how many messages the visitor has sent
user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
limit_reached = user_turns >= MESSAGE_LIMIT


def submit(text):
    st.session_state.pending = text


# Suggestion chips (clickable) — hidden while email capture is showing
if (st.session_state.chips and not limit_reached
        and st.session_state.pending is None
        and not st.session_state.show_email_capture):
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
            answer, chips, status = get_response(st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chips = chips
    # Show the email capture only when the bot is genuinely unavailable
    st.session_state.show_email_capture = (status == "unavailable")
    st.rerun()

# ── Feedback ──
# The feedback form opens via the "💬 Feedback" button at the top, available
# ── Feedback form lives in the sidebar (see sidebar block above) ──

# Show the thank-you as a brief toast (auto-dismisses, doesn't linger)
if st.session_state.get("feedback_just_saved"):
    st.toast("Thank you for the feedback! Pratham appreciates it. 🙏", icon="🙏")
    st.session_state.feedback_just_saved = False