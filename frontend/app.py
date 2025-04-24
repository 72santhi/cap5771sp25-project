import sys, os
import streamlit as st

# so Python can find source_code/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source_code.recommender import recommend_next_items, item_vocab

st.set_page_config(page_title="Session-Based Recommender", layout="wide")

# Pre-split the vocab by prefix
movie_items = sorted([i for i in item_vocab if i.startswith("Movie_")])
music_items = sorted([i for i in item_vocab if i.startswith("Music_")])
book_items  = sorted([i for i in item_vocab if i.startswith("Book_")])

# Session clicks
if "clicks" not in st.session_state:
    st.session_state.clicks = []

st.title("Session-Based Cross-Domain Recommender")
st.markdown("**Motto:** Your next picks follow your **current** clicks.")

with st.sidebar:
    st.header("🔎 Add a Click")
    # 1) pick domain
    domain = st.radio("Category", ["Movie", "Music", "Book"])
    # 2) pick item from that domain
    if domain == "Movie":
        choice = st.selectbox("Select a Movie", movie_items)
    elif domain == "Music":
        choice = st.selectbox("Select a Music Track", music_items)
    else:
        choice = st.selectbox("Select a Book", book_items)

    if st.button("➕ Add to Session"):
        st.session_state.clicks.append(choice)
    if st.button("🔄 Reset Session"):
        st.session_state.clicks = []

st.subheader("Your Session Clicks")
st.write(st.session_state.clicks or "_No clicks yet._")

if st.session_state.clicks:
    st.subheader("Top-5 Recommendations")
    for i, rec in enumerate(recommend_next_items(st.session_state.clicks, top_k=5), 1):
        st.write(f"{i}. {rec}")
