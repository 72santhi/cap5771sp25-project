# 🎬 Session-Based Cross-Domain Recommender

An interactive recommendation system that predicts the next best item (Movie, Music, or Book) based on a user's ongoing session using a GRU4Rec model. The system is deployed via a Streamlit dashboard with real-time interaction.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)
![License](https://img.shields.io/github/license/72santhi/cap5771sp25-project)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Project Overview

- **Model**: GRU4Rec (session-based RNN)
- **Domains**: Movies, Music, Books
- **Datasets**: IMDb, Spotify, Goodreads
- **Features**:
  - 🔁 Session tracking  
  - 🔮 Live top-5 predictions    

---


## 📺 Demo Videos

## 📺 Demo Videos

### 🔹 Dashboard Demo (4 mins)
<iframe width="640" height="360"
  src="https://youtu.be/w1nElGVKB1Y"
  frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
</iframe>

### 🔹 Dataset & Model Walkthrough (4 mins)
<iframe width="640" height="360"
  src="https://youtu.be/9GcDJNJSX6Q"
  frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
</iframe>


## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/72santhi/cap5771sp25-project.git
cd cap5771sp25-project

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Prepare model weights
 Option A: Train locally
    Run:
    source_code/Model_Training_&_Evaluation.ipynb
    Then ensure(move the .pth file to source_code/models dir):
    source_code/models/best_session_rec_model.pth exists

 Option B: Download pretrained weights
    Download from:
    https://drive.google.com/file/d/1wLrek6zZWnobHuXkVKmaQmbUgEWBZZxK/view?usp=sharing
    Then:
mkdir -p source_code/models
mv best_session_rec_model.pth source_code/models/

# 4. Launch the dashboard
streamlit run frontend/app.py

# 5. Open in browser:
    http://localhost:8501

# Project Structure
cap5771sp25-project/
├── .streamlit/
│   ├── config.toml
|   ├── session_events.csv
├── Data/
│   └── data_access_info.txt          
├── frontend/
│   └── app.py                         # Streamlit UI
├── source_code/
  |   ├──__init__.py 
│   ├── recommender.py                 # GRU4Rec inference code
│   ├── session_events.csv             # Simulated session data
│   ├── Model_Training_&_Evaluation.ipynb
|   ├── Data_Preprocessing_&_Visualizations
│   └── models/
│       └── best_session_rec_model.pth
├── Reports/
│   ├── Milestone1.pdf
|   ├── Milestone2.pdf
|   ├── Milestone3.pdf
|   └── Video Demo/
|       ├── Dashboard Demo.mpv4
|       ├── Cross_DOmain_Recomondation_System_approach.mpv4
├── requirements.txt
└── README.md

```
