import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import json
import logging
from backend.services.demo_service import ResearchDemonstratorService

# Configure basic logging if not already done
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="H.E.A.R.T. AI Companion",
    layout="centered"
)

# Initialize the service globally (cached)
@st.cache_resource
def get_service():
    return ResearchDemonstratorService.get_instance()

service = get_service()

if "history" not in st.session_state:
    st.session_state.history = []

# --- Minimal CSS ---
st.markdown("""
<style>
.stApp {
    font-family: 'Inter', sans-serif;
}
.companion-title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0px;
}
.companion-subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2rem;
    margin-top: 0px;
    margin-bottom: 30px;
}
.chat-bubble {
    background-color: #f1f8ff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 4px solid #3498db;
    color: #333;
    font-size: 1.1rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='companion-title'>H.E.A.R.T. AI</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Conversation (MP4 only)", type=["mp4"])

if uploaded_file is not None:
    if st.button("Analyze Conversation", type="primary"):
        with st.spinner("H.E.A.R.T. AI is listening..."):
            try:
                video_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
                result = service.analyze_conversation(
                    mode="upload",
                    video_bytes=video_bytes,
                    video_filename=filename,
                    history=st.session_state.history,
                    text="" 
                )
                
                comp_dict = result.get("companion_response", {})
                if isinstance(comp_dict, dict):
                    companion_response = comp_dict.get("reply", "")
                    transcript = comp_dict.get("user_utterance", result.get("utterance_text", ""))
                else:
                    companion_response = str(comp_dict)
                    transcript = result.get("utterance_text", "")
                    
                emotion = result.get("fusion", {}).get("predicted_emotion", "Unknown").capitalize()
                rel_level = result.get("reliability", {}).get("level", "Unknown")
                
                if not transcript or str(transcript).strip() == "":
                    transcript = "(No speech detected)"
                    
                st.session_state.history.append({"role": "user", "content": transcript})
                st.session_state.history.append({"role": "companion", "content": companion_response})

                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]
                    
                st.success("Analysis Complete")
                st.markdown(f"**I heard:**<br/> _{transcript}_", unsafe_allow_html=True)
                st.markdown(f"**How you're feeling:** {emotion}", unsafe_allow_html=True)
                
                rel_msg = ""
                if rel_level == "HIGH":
                    rel_msg = "I feel fairly confident about this."
                elif rel_level == "MEDIUM":
                    rel_msg = "I'm not completely sure, so I'll keep my response gentle."
                else:
                    rel_msg = "I'm having trouble reading the situation, so I'll focus on clarification."
                
                st.markdown(f"<div style='color: #7f8c8d; font-size: 0.9em; margin-bottom: 10px; margin-top: 10px;'>{rel_msg}</div>", unsafe_allow_html=True)
                
                st.markdown("### H.E.A.R.T. AI:")
                st.markdown(f"<div class='chat-bubble'>{companion_response}</div>", unsafe_allow_html=True)
                
                with st.expander("Why did H.E.A.R.T. AI respond this way? (Analysis Details)"):
                    conf = result.get("fusion", {}).get("confidence_calibrated", 0.0)
                    d1 = result.get("disagreement", {}).get("d1", 0.0)
                    d3 = result.get("disagreement", {}).get("d3", 0)
                    strategy = result.get("response_policy", {}).get("adaptation_name", "Unknown")
                    st.write(f"**Fused Emotion:** {emotion}")
                    st.write(f"**Confidence:** {conf*100:.1f}%")
                    st.write(f"**Cross-modal JSD (D1):** {d1:.3f}")
                    st.write(f"**Categorical Conflict (D3):** {d3}")
                    st.write(f"**Reliability Level:** {rel_level}")
                    st.write(f"**Response Strategy:** {strategy}")
                
            except Exception as e:
                st.error(f"Error analyzing conversation: {str(e)}")
                logger.error("Analysis error", exc_info=True)