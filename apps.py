from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np
import streamlit as st
import pandas as pd
import papermill as pm
import os
import base64

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# -----------------------------
# FUNCTION TO LOAD IMAGES
# -----------------------------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    return ""

logo_base64 = get_base64_image("logo.png")
book_bg_base64 = get_base64_image("open_book.png")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(f"""
<style>

/* Background */
.stApp {{
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}}

/* Navbar */
.navbar {{
    background: cornflowerblue;
    padding: 12px 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
}}

.navbar img {{
    width: 42px;
    height: 42px;
    margin-right: 12px;
    border-radius: 50%;
    border: 2px solid white;
}}

.navbar-title {{
    font-size: 22px;
    font-weight: bold;
    color: white;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: #B0E0E6;
}}

/* Moving Quotes */
.marquee {{
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    margin-bottom: 10px;
}}

.marquee span {{
    display: inline-block;
    padding-left: 100%;
    animation: marquee 18s linear infinite;
    font-size: 24px;
    font-weight: bold;
    color: gold;
}}

@keyframes marquee {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-100%); }}
}}

/* Titles */
.main-title {{
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #FFD700;
}}

.sub-text {{
    text-align: center;
    font-size: 1.2rem;
    color: #f0f0f0;
}}

/* Hero Section */
.hero {{
    background: linear-gradient(rgba(15,32,39,0.85), rgba(44,83,100,0.9)),
                url("data:image/png;base64,{book_bg_base64}");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    padding: 80px 20px;
    border-radius: 20px;
    text-align: center;
}}

/* Cards */
.card {{
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}}

.card:hover {{
    transform: translateY(-8px);
}}

/* Book Cards */
.book-card {{
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    transition: 0.3s;
    height: 360px;
}}

.book-card img {{
    height: 180px;
    object-fit: contain;
}}

.book-card:hover {{
    transform: scale(1.05);
}}

/* Search Label */
div[data-testid="stTextInput"] label {{
    color: white !important;
    font-weight: bold;
    font-size: 18px;
}}

/* Search Input */
.stTextInput input {{
    background-color: rgba(255,255,255,0.95);
    color: black !important;
    border-radius: 10px;
    padding: 10px;
}}

/* Footer */
.footer {{
    text-align: center;
    padding: 15px;
    margin-top: 80px;
    font-size: 14px;
    color: #ddd;
    border-top: 1px solid rgba(255,255,255,0.2);
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVBAR
# -----------------------------
if logo_base64:
    st.markdown(f"""
    <div class="navbar">
        <img src="data:image/png;base64,{logo_base64}">
        <div class="navbar-title">AI Book Recommendation System</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="navbar">
        <div class="navbar-title">📚 AI Book Recommendation System</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📖 Navigation")
menu = ["Home", "About", "Search Book"]
choice = st.sidebar.radio("BOOK RECOMMENDATION NAVBAR", menu)
st.session_state.page = choice

# -----------------------------
# BUILD MODEL IF NOT EXISTS
# -----------------------------
if not (os.path.exists('Dataset/final_data.csv') and os.path.exists('Model/cosine_sim.npy')):
    st.warning('⚠️ Models not found! Generating models... Please wait.')
    pm.execute_notebook(
        'Model/recommendation_model.ipynb',
        'output_notebook.ipynb'
    )
    st.success("✅ Model generation completed successfully!")
else:
    st.info("System initialized successfully...")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource()
def load_models():
    cosine_sim = np.load('Model/cosine_sim.npy')
    ncf_model = load_model('Model/ncf_model.h5')
    cnn_model = load_model("Model/cnn_model.h5")
    df = pd.read_csv("Dataset/final_data_with_ratings.csv")

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['Desc'].astype(str))

    return cosine_sim, df, ncf_model, cnn_model, tokenizer

cosine_sim, final_data, ncf_model, cnn_model, tokenizer = load_models()

# -----------------------------
# HYBRID RECOMMENDATION FUNCTION (UNCHANGED)
# -----------------------------
def hybrid_recommendation(book_title, df, cosine_sim, ncf_model, cnn_model, tokenizer, max_len=200):

    def recommend_books_cosine(book_title, final_data, cosine_sim):
        matches = final_data[
            final_data['Title'].str.contains(book_title, case=False, na=False)
        ]

        if matches.empty:
            return None

        idx = matches.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        book_indices = [i[0] for i in sim_scores]

        return final_data[['Title', 'Image', 'Author', 'Pages', 'ISBN', 'Desc']].iloc[book_indices]

    cosine_recs = recommend_books_cosine(book_title, df, cosine_sim)

    if cosine_recs is None:
        st.error("❌ No matching books found. Try a different title.")
        return None

    user_id = df['user_id'].iloc[0]

    book_ids = [
        df[df['Title'] == title]['ISBN'].values[0]
        for title in cosine_recs['Title']
    ]

    ncf_recs = []
    for book_id in book_ids:
        user_tensor = torch.tensor([int(user_id)], dtype=torch.long)
        book_tensor = torch.tensor([int(book_id)], dtype=torch.long)
        rating_pred = ncf_model([user_tensor, book_tensor])[0].numpy()
        ncf_recs.append((book_id, rating_pred))

    ncf_recs = sorted(ncf_recs, key=lambda x: x[1], reverse=True)

    cnn_recs = []
    for book_id in book_ids:
        book_desc = df[df['ISBN'] == book_id]['Desc'].values[0]
        book_seq = tokenizer.texts_to_sequences([book_desc])
        book_pad = pad_sequences(book_seq, maxlen=max_len)
        rating_pred = cnn_model.predict(book_pad, verbose=0).item()
        cnn_recs.append((book_id, rating_pred))

    cnn_recs = sorted(cnn_recs, key=lambda x: x[1], reverse=True)

    combined_books = list(set([book for book, _ in cnn_recs]))

    final_recs = [
        (
            df[df['ISBN'] == book_id]['Title'].values[0],
            df[df['ISBN'] == book_id]['Image'].values[0],
            df[df['ISBN'] == book_id]['Author'].values[0],
            df[df['ISBN'] == book_id]['Pages'].values[0]
        )
        for book_id in combined_books
    ]

    return pd.DataFrame(final_recs, columns=['Title', 'Image', 'Author', 'Pages'])

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="marquee">
        <span>
        📚 ""Books are mirrors: you only see in them what you already have inside you." – Carlos Ruiz Zafón &nbsp;&nbsp;&nbsp;
        📖 "Today a reader, tomorrow a leader." – Margaret Fuller &nbsp;&nbsp;&nbsp;
        ✨ Discover. Learn. Grow. Read.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1 class="main-title">Welcome to Smart Book Recommender</h1>
        <p class="sub-text">
            Discover your next favorite book with intelligent recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">📚<h3>Personalized Picks</h3><p>Recommendations tailored to your interests.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">🌍<h3>Extensive Library</h3><p>Explore books across genres and authors.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">⚡<h3>Quick Discovery</h3><p>Find your next read in seconds.</p></div>', unsafe_allow_html=True)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif st.session_state.page == "About":
    st.markdown("""
    <div class="hero">
        <h2 class="main-title">About This System</h2>
        <p class="sub-text">
        Books are gateways to knowledge, imagination, and inspiration. This intelligent
        book recommendation system helps readers discover meaningful and engaging titles
        based on their interests. By analyzing book titles and descriptions, it suggests
        similar books that align with a reader’s preferences. Whether you are a student,
        researcher, or passionate reader, this platform makes exploring literature simple,
        enjoyable, and rewarding.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SEARCH BOOK PAGE
# -----------------------------
elif st.session_state.page == "Search Book":
    st.markdown('<h2 class="main-title">🔍 Search Books</h2>', unsafe_allow_html=True)

    selected_option = st.text_input(
        "Enter the title or content",
        placeholder="Try: Harry Potter, Hobbit, Atomic Habits..."
    )

    if selected_option:
        with st.spinner("🔄 Finding best recommendations..."):
            book = hybrid_recommendation(
                selected_option,
                final_data,
                cosine_sim,
                ncf_model,
                cnn_model,
                tokenizer
            )

        if book is not None and len(book) > 0:
            st.subheader("📖 Recommended Books")
            cols = st.columns(5)

            for i in range(min(5, len(book))):
                with cols[i]:
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{book.iloc[i,1]}">
                        <br><br>
                        <b>{book.iloc[i,0]}</b><br>
                        ✍️ {book.iloc[i,2]}<br>
                        📄 {book.iloc[i,3]} pages
                    </div>
                    """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    © 2026 AI Book Recommendation System | Developed by <b>Liya Baby</b> ❤️ <br>
    Empowering Readers Through Intelligent Recommendations
</div>
""", unsafe_allow_html=True)