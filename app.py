import streamlit as st
import time
import base64
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="MeetMind AI | Intelligence Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enhanced CSS (Glassmorphism & Cyberpunk) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Syne:wght@700;800&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --primary: #ff6b35;
    --secondary: #7209b7;
    --bg-dark: #050508;
    --card-bg: rgba(20, 20, 35, 0.7);
    --border: rgba(255, 107, 53, 0.2);
}

/* Base Styles */
.stApp {
    background: radial-gradient(circle at top right, #1a1a2e, #050508);
    color: #e8e4dc;
    font-family: 'Inter', sans-serif;
}

/* Hide Default Elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 5% !important; }

/* Hero Section */
.hero-container {
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 0.9;
    background: linear-gradient(90deg, #fff 0%, var(--primary) 50%, #ff9500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Glass Cards */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.glass-card:hover {
    border-color: var(--primary);
    box-shadow: 0 0 20px rgba(255, 107, 53, 0.15);
}

/* Input Overrides */
.stTextInput input, .stSelectbox select {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid var(--border) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Fancy Buttons */
div.stButton > button {
    background: linear-gradient(90deg, var(--primary), #ff9500) !important;
    color: white !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
}

/* Checklist Styling */
.check-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #ccc;
}
.check-icon { color: var(--primary); font-weight: bold; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">MEETMIND.AI</div>
    <div style="font-family:'JetBrains Mono'; color: #666; letter-spacing: 3px; font-size: 0.8rem;">
        COGNITIVE VIDEO ANALYSIS // RAG v2.0
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar/Inputs ---
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        source = st.text_input("Data Source (YouTube URL or File Path)", placeholder="Paste link here...")
    with c2:
        language = st.selectbox("Intelligence Mode", ["english", "hinglish"])
    
    run_btn = st.button("⚡ ANALYZE NEURAL STREAM")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Logic Processing ---
if run_btn and source:
    # Reset state for new run
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []
    
    steps = [
        ("🔊", "Processing Audio"),
        ("✍️", "Transcribing"),
        ("🏷️", "Defining Title"),
        ("📋", "Synthesizing Summary"),
        ("✅", "Mapping Actions"),
        ("🔑", "Extracting Decisions"),
        ("❓", "Isolating Queries"),
        ("🧠", "Indexing Knowledge"),
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # We simulate visual progress while running tasks
        for i, (icon, msg) in enumerate(steps):
            status_text.markdown(f"**{icon} System Status:** {msg}...")
            progress_bar.progress((i + 1) / len(steps))
            
            # Actual execution mapped to steps
            if i == 0: chunks = process_input(source)
            elif i == 1: transcript = transcribe_all(chunks, language)
            elif i == 2: title = generate_title(transcript)
            elif i == 3: summary = summarize(transcript)
            elif i == 4: action_items = extract_action_items(transcript)
            elif i == 5: decisions = extract_key_decisions(transcript)
            elif i == 6: questions = extract_questions(transcript)
            elif i == 7: rag_chain = build_rag_chain(transcript)

        st.session_state.pipeline_result = {
            "title": title, "transcript": transcript, "summary": summary,
            "action_items": action_items, "key_decisions": decisions,
            "open_questions": questions, "rag_chain": rag_chain
        }
        st.success("Neural Analysis Complete.")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"Critical System Failure: {e}")

# --- Results Rendering ---
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result

    # 1. Title Highlight
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid var(--primary);">
        <small style="color:var(--primary); font-family:'JetBrains Mono';">SESSION TITLE</small>
        <h2 style="margin:0; font-family:'Syne';">{res['title']}</h2>
    </div>
    """, unsafe_allow_html=True)

    # 2. Summary and Transcript Split
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:var(--primary); font-family:'Syne'; font-size:0.9rem;">📋 EXECUTIVE SUMMARY</h4>
            <p style="font-size:0.95rem; line-height:1.6; color:#ddd;">{res['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:var(--primary); font-family:'Syne'; font-size:0.9rem;">📄 RAW DATA</h4>
            <div style="height:200px; overflow-y:auto; font-family:'JetBrains Mono'; font-size:0.75rem; color:#888;">
                {res['transcript']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Triple Grid (Actions, Decisions, Questions)
    c1, c2, c3 = st.columns(3)
    
    def format_list(text, icon):
        items = [i.strip() for i in text.split('\n') if i.strip()]
        html = ""
        for item in items:
            clean = item.lstrip("-•*123456789.) ")
            html += f'<div class="check-item"><span class="check-icon">{icon}</span>{clean}</div>'
        return html

    with c1:
        st.markdown(f'<div class="glass-card"><h4 style="font-size:0.8rem; color:var(--primary);">ACTION ITEMS</h4>{format_list(res["action_items"], "→")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-card"><h4 style="font-size:0.8rem; color:#7209b7;">KEY DECISIONS</h4>{format_list(res["key_decisions"], "✦")}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="glass-card"><h4 style="font-size:0.8rem; color:#ff9500;">OPEN QUESTIONS</h4>{format_list(res["open_questions"], "?")}</div>', unsafe_allow_html=True)

    # 4. RAG Chat Section
    st.markdown("<br><h3 style='font-family:Syne;'>💬 NEURAL CHAT ASSISTANT</h3>", unsafe_allow_html=True)
    
    # Custom Chat UI Container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant", avatar="🧠").write(message["content"])

    if prompt := st.chat_input("Query the meeting memory..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving from context..."):
                response = ask_question(res["rag_chain"], prompt)
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear Chat
    if st.session_state.chat_history:
        if st.button("Wipe Chat Memory"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty State
    st.markdown("""
    <div style="text-align:center; padding: 100px 0; opacity: 0.3;">
        <h1 style="font-size: 100px; margin:0;">📡</h1>
        <p style="font-family:'JetBrains Mono';">AWAITING INPUT STREAM...</p>
    </div>
    """, unsafe_allow_html=True)