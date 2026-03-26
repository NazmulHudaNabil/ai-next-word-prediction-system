import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(
    page_title="Next Word Predictor — LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── Custom CSS ──────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0a1a;
    --bg-card: rgba(255,255,255,0.04);
    --accent: #6c63ff;
    --accent-light: #a29bfe;
    --text-primary: #f0f0f5;
    --text-muted: #8888aa;
    --glow: 0 0 30px rgba(108,99,255,0.25);
    --radius: 16px;
}

/* ── Global overrides ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #111133 50%, #0a0a1a 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0e2a 0%, #14143a 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.15);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hero ── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 999px;
    padding: 6px 20px;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent-light);
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #a29bfe 50%, #6c63ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
    line-height: 1.15;
}

.hero-sub {
    font-size: 1.1rem;
    color: var(--text-muted);
    max-width: 580px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 40px rgba(108,99,255,0.35);
}

/* ── Prediction card ── */
.prediction-word {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(162,155,254,0.10));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 14px 24px;
    margin: 6px;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-light);
    cursor: default;
    transition: all 0.3s ease;
}

.prediction-word:hover {
    background: linear-gradient(135deg, rgba(108,99,255,0.30), rgba(162,155,254,0.20));
    transform: scale(1.05);
}

.prediction-rank {
    font-size: 0.72rem;
    font-weight: 700;
    background: var(--accent);
    color: #fff;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

/* ── Probability bar ── */
.prob-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
    margin-top: 6px;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent-light));
    transition: width 0.6s ease;
}

/* ── Result table ── */
.result-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.result-row:last-child { border-bottom: none; }

.result-word {
    min-width: 120px;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text-primary);
}

.result-prob {
    font-size: 0.85rem;
    color: var(--text-muted);
    min-width: 60px;
    text-align: right;
}

/* ── Completed sentence card ── */
.sentence-output {
    font-size: 1.25rem;
    font-weight: 600;
    color: #fff;
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(162,155,254,0.08));
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: var(--radius);
    line-height: 1.7;
    word-spacing: 3px;
    letter-spacing: 0.3px;
}

.sentence-output .predicted {
    color: var(--accent-light);
    text-decoration: underline;
    text-underline-offset: 4px;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Input styling ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(108,99,255,0.25) !important;
    border-radius: 12px !important;
    color: #fff !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(108,99,255,0.2) !important;
}

[data-testid="stTextInput"] label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #5a52e0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,99,255,0.4) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] span {
    color: var(--text-muted) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 3rem;
}

.footer-heart { color: #ff6b6b; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeInUp 0.6s ease forwards;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────── Load Model & Tokenizer ─────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the LSTM model, tokenizer, and max sequence length."""
    model = load_model("LSTM Model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("Model max len.pkl", "rb") as f:
        max_len = pickle.load(f)
    # Use the model's actual input shape for padding
    input_len = model.input_shape[1]  # e.g. 745
    return model, tokenizer, max_len, input_len


def predict_next_words(model, tokenizer, input_len, text, top_k=5):
    """Return top-k predicted words with probabilities."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return []

    token_list = token_list[-input_len:]
    token_list = pad_sequences([token_list], maxlen=input_len, padding="pre")

    probs = model.predict(token_list, verbose=0)[0]
    top_indices = probs.argsort()[-top_k:][::-1]

    reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}
    results = []
    for idx in top_indices:
        word = reverse_word_map.get(idx, None)
        if word:
            results.append((word, float(probs[idx])))
    return results


def predict_sequence(model, tokenizer, input_len, text, n_words=1):
    """Generate n words one at a time and return the full sentence."""
    current = text
    for _ in range(n_words):
        preds = predict_next_words(model, tokenizer, input_len, current, top_k=1)
        if not preds:
            break
        current += " " + preds[0][0]
    return current


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    loading_msg = st.empty()
    loading_msg.info("Loading model…")

    model, tokenizer, max_len, input_len = load_artifacts()
    vocab_size = len(tokenizer.word_index) + 1

    loading_msg.success("Model loaded ✓")

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">LSTM</div>
            <div class="stat-label">Architecture</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{vocab_size:,}</div>
            <div class="stat-label">Vocabulary</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{max_len}</div>
            <div class="stat-label">Max Length</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K predictions", 1, 10, 5)
    n_words = st.slider("Words to generate", 1, 20, 3)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:var(--text-muted);line-height:1.7;">
        <strong style="color:#a29bfe;">How it works</strong><br>
        The model reads your input, tokenizes it, and uses an LSTM neural network
        to predict the most probable next word based on learned language patterns.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────── Hero Section ────────────────────────
st.markdown("""
<div class="hero-container animate-in">
    <div class="hero-badge">⚡ Deep Learning Powered</div>
    <div class="hero-title">Next Word Predictor</div>
    <p class="hero-sub">
        Type a sentence and let our LSTM neural network predict
        what comes next — with confidence scores for every suggestion.
    </p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────── Input Area ──────────────────────────
col_input, col_btn = st.columns([4, 1])

with col_input:
    seed_text = st.text_input(
        "YOUR TEXT",
        placeholder="e.g.  I am going to the …",
        label_visibility="visible",
    )

with col_btn:
    st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)  # align button
    predict_btn = st.button("🚀 Predict", use_container_width=True)


# ───────────────────────── Results ─────────────────────────────
if predict_btn and seed_text.strip():
    with st.spinner("Thinking…"):
        predictions = predict_next_words(model, tokenizer, input_len, seed_text, top_k)
        full_sentence = predict_sequence(model, tokenizer, input_len, seed_text, n_words)

    if predictions:
        # ── Generated Sentence ──
        st.markdown("#### ✨ Generated Sentence")
        generated_part = full_sentence[len(seed_text):].strip()
        st.markdown(f"""
        <div class="glass-card sentence-output animate-in">
            {seed_text} <span class="predicted">{generated_part}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Top Predictions ──
        st.markdown("#### 🏆 Top Predictions")
        st.markdown('<div class="glass-card animate-in">', unsafe_allow_html=True)

        max_prob = max(p for _, p in predictions) if predictions else 1

        for rank, (word, prob) in enumerate(predictions, 1):
            bar_width = (prob / max_prob) * 100
            st.markdown(f"""
            <div class="result-row">
                <span class="prediction-rank">{rank}</span>
                <span class="result-word">{word}</span>
                <div style="flex:1">
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                </div>
                <span class="result-prob">{prob:.2%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Quick word chips ──
        st.markdown("#### 🔮 Quick Picks")
        chips_html = "".join(
            f'<span class="prediction-word animate-in"><span class="prediction-rank">{i}</span>{w}</span>'
            for i, (w, _) in enumerate(predictions, 1)
        )
        st.markdown(f'<div style="text-align:center">{chips_html}</div>', unsafe_allow_html=True)

    else:
        st.warning("Could not predict — the input may not contain recognizable words.")

elif predict_btn:
    st.warning("Please enter some text first.")


# ───────────────────────── Footer ──────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span class="footer-heart">♥</span> using Streamlit & TensorFlow  •  LSTM Next Word Predictor
</div>
""", unsafe_allow_html=True)
