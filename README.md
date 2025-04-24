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
   <br>![Demo Video](https://drive.google.com/file/d/1C1WuEWpatL2zhSvaX_6W_M98UIe9VejN/view?usp=sharing)  
   *(A 4-minute walkthrough of the Streamlit app: session building & recommendations.)*

2. **Approach & Dataset Overview**  
   <br>![Approach Video](https://drive.google.com/file/d/1Iag5igRfxkv7yLTSvamyQ1tScOKBfdf2/view?usp=drive_link)  
   *(A 4-minute summary of data preprocessing, model architecture, and evaluation.)*

---

## 🚀 Installation

1. **Clone the repo** (private):  
   ```bash
   git clone https://github.com/<username>/cap5771sp25-project.git
   cd cap5771sp25-project

1. **download the model** (private):
   You can run the https://github.com/72santhi/cap5771sp25-project/blob/main/source_code/Model_Training_%26_Evaluation.ipynb and store the best_session_rec_model.pth" in \source_scoe\model directory

   or

   Download the model fromt he link and place in the \source_scoe\model directory

