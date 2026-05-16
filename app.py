import streamlit as st
import time
from dotenv import load_dotenv

# 1. LOAD ENV VARS FIRST
load_dotenv()

# 2. CUSTOM MODULES
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System (Refined CSS) ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@300;400;500&display=swap');

/* Apply custom fonts */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
code, pre, .stChatMessage {
    font-family: 'Fira Code', monospace !important;
}

/* Beautiful Gradient Text for Main Headers */
.gradient-text {
    background: -webkit-linear-gradient(45deg, #e8440a, #2d6ef6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0px;
}

/* Custom Cards for Data */
.mm-card {
    background: #ffffff;
    border: 1px solid #e2e0d8;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 15px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.mm-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.dark-mode .mm-card {
    background: #1e1e22;
    border: 1px solid #333;
}

/* Extraction Lists */
.mm-list { list-style: none; padding: 0; margin: 0; }
.mm-list li {
    padding: 12px;
    margin-bottom: 8px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.5;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.mm-action { background: rgba(232,68,10,0.05); border-left: 3px solid #e8440a; }
.mm-decision { background: rgba(45,110,246,0.05); border-left: 3px solid #2d6ef6; }
.mm-question { background: rgba(26,184,122,0.05); border-left: 3px solid #1ab87a; }

/* Transcript Scroll Box */
.mm-transcript-box {
    height: 400px;
    overflow-y: auto;
    background: #f4f3ef;
    padding: 20px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.8;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Meet*Mind*")
    st.caption("COGNITIVE PIPELINE • RAG v2.0")
    st.divider()
    
    st.markdown("### Workspace")
    st.button("⚡ Analyze New", use_container_width=True, type="primary")
    st.button("📁 Sessions (3)", use_container_width=True)
    st.button("📚 Library", use_container_width=True)
    
    st.divider()
    st.markdown("### System Specs")
    st.code("Model: mistral-small-latest\nAudio: whisper-small\nVector: Chroma\nEmbed: all-MiniLM-L6", language="yaml")
    st.success("🟢 System Ready")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="gradient-text">Neural Stream Ingestion</p>', unsafe_allow_html=True)
st.write("Convert any meeting into structured intelligence, decisions, and searchable memory.")
st.markdown("<br>", unsafe_allow_html=True)

# ── Input Area ────────────────────────────────────────────────────────────────
with st.container(border=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        source = st.text_input("Source URL or Path", placeholder="https://youtube.com/watch?v=... or /path/to/meeting.mp4", label_visibility="collapsed")
    with col2:
        language = st.selectbox("Language", ["english", "hinglish"], label_visibility="collapsed")

    run_btn = st.button("⚡ Initiate Analysis", type="primary", use_container_width=True)

# ── Pipeline Execution (Native Streamlit Status) ──────────────────────────────
if run_btn and source:
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []
    
    # st.status provides a gorgeous native loading state
    with st.status("Initializing Neural Pipeline...", expanded=True) as status:
        try:
            st.write("🔊 Extracting and processing audio chunks...")
            chunks = process_input(source)
            
            st.write("✍️ Transcribing with Whisper...")
            transcript = transcribe_all(chunks, language)
            
            st.write("🏷 Generating semantic title...")
            title = generate_title(transcript)
            
            st.write("📋 Compiling executive summary...")
            summary = summarize(transcript)
            
            st.write("🎯 Extracting action items, decisions, and questions...")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            
            st.write("🧠 Indexing vectors for RAG...")
            rag_chain = build_rag_chain(transcript)

            st.session_state.pipeline_result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            st.rerun()

        except Exception as e:
            status.update(label="Pipeline Failure", state="error", expanded=True)
            st.error(f"Error details: {e}")

# ── Results Area ──────────────────────────────────────────────────────────────
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    
    # Header & Metrics
    st.markdown(f"## {res['title']}")
    
    def count_items(text):
        return len([l for l in text.strip().split("\n") if l.strip()])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words Indexed", f"{len(res['transcript'].split()):,}")
    m2.metric("Action Items", count_items(res["action_items"]))
    m3.metric("Key Decisions", count_items(res["key_decisions"]))
    m4.metric("Est. Duration", f"{len(res['transcript'].split()) // 130} min")
    
    st.divider()

    # Tabs for clean UX
    tab_summary, tab_extracts, tab_transcript, tab_chat = st.tabs([
        "📋 Executive Summary", 
        "🎯 Key Extractions", 
        "📄 Transcript", 
        "🧠 Neural Chat"
    ])

    with tab_summary:
        st.markdown(f"""
        <div class="mm-card">
            <h4>Executive Overview</h4>
            <p style="font-size: 1.1rem; line-height: 1.8;">{res["summary"]}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab_extracts:
        col_act, col_dec, col_que = st.columns(3)
        
        def render_html_list(text, css_class):
            lines = [l.strip().lstrip("-•*0123456789.) ").strip() for l in text.strip().split("\n") if l.strip()]
            if not lines: return "<p>None found.</p>"
            list_items = "".join([f"<li class='{css_class}'>{line}</li>" for line in lines])
            return f"<ul class='mm-list'>{list_items}</ul>"

        with col_act:
            st.markdown("#### ✅ Actions")
            st.markdown(render_html_list(res["action_items"], "mm-action"), unsafe_allow_html=True)
        with col_dec:
            st.markdown("#### 🔑 Decisions")
            st.markdown(render_html_list(res["key_decisions"], "mm-decision"), unsafe_allow_html=True)
        with col_que:
            st.markdown("#### ❓ Open Questions")
            st.markdown(render_html_list(res["open_questions"], "mm-question"), unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(f'<div class="mm-transcript-box">{res["transcript"]}</div>', unsafe_allow_html=True)

    with tab_chat:
        st.markdown("### Query the Meeting Context")
        
        # Chat container to keep it bound within the tab
        chat_container = st.container(height=400)
        
        with chat_container:
            for msg in st.session_state.chat_history:
                avatar = "👤" if msg["role"] == "user" else "🧠"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])

        if prompt := st.chat_input("Ask anything about this meeting..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.write(prompt)
                
                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Searching neural context..."):
                        answer = ask_question(res["rag_chain"], prompt)
                        st.write(answer)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

elif not run_btn:
    # Empty State
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px; color: #8a8880;">
        <h1 style="font-size: 4rem; opacity: 0.2; margin-bottom: 0;">📡</h1>
        <h3 style="font-family: 'Fira Code', monospace; letter-spacing: 2px; text-transform: uppercase; font-size: 1rem;">Awaiting Data Stream</h3>
        <p>Enter a meeting source above to begin cognitive processing.</p>
    </div>
    """, unsafe_allow_html=True)