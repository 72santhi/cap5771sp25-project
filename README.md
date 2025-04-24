# Session-Based Cross-Domain Recommender

A demo of a GRU4Rec-powered recommendation engine that suggests next items (movies, music, books) based on your current click session. Packed into an interactive Streamlit dashboard with Netflix-style theming.

---

## 🎯 Project Overview

- **Model**: GRU4Rec (session-based RNN)  
- **Domains**: Movies, Music, Books  
- **Data Sources**:  
  - IMDb Movie Dataset  
  - Spotify Tracks Dataset  
  - Goodreads Book Metadata  
- **Key Capabilities**:  
  1. **Live Session Simulation**: Click domain-segmented items and see top-5 next-item recommendations in real time.  
  2. **Performance Metrics**: Precision@K, Recall@K, F1@K plotted over training epochs.  
  3. **Bias Analysis**: Popularity & domain imbalance explorer (planned).  
  4. **Dark “Netflix” Theme**: Pitch-black background, red accents, white text.

---

## 📺 Demo Videos

1. **Live Dashboard Demo**  
   <br>![Demo Video](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)  
   *(A 4-minute walkthrough of the Streamlit app: session building, recommendations, theming.)*

2. **Approach & Dataset Overview**  
   <br>![Approach Video](https://www.youtube.com/watch?v=APPROACH_VIDEO_ID)  
   *(A 2-minute summary of data preprocessing, model architecture, and evaluation plan.)*

---

## 🚀 Installation

1. **Clone the repo** (private):  
   ```bash
   git clone https://github.com/<username>/cap5771sp25-project.git
   cd cap5771sp25-project
