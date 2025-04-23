# source_code/recommender.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn

# ─── BASE DIR ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

# ─── MODEL ──────────────────────────────────────────────────────────────────────
class GRU4Rec(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x, _ = self.gru(self.embedding(x))
        return self.fc(x)

# ─── LOAD CHECKPOINT & BUILD VOCAB ──────────────────────────────────────────────
DEVICE     = torch.device("cpu")
CKPT_PATH  = os.path.join(BASE_DIR, "models", "best_session_rec_model.pth")
CSV_PATH   = os.path.join(BASE_DIR, "session_events.csv")

ckpt       = torch.load(CKPT_PATH, map_location=DEVICE)
state_dict = ckpt.get("model_state", ckpt)
vocab_size = state_dict["embedding.weight"].shape[0]

# Read your CSV once, take first `vocab_size` unique items
df          = pd.read_csv(CSV_PATH)
unique_ids  = df["Item_ID"].drop_duplicates().tolist()
unique_ids  = unique_ids[:vocab_size]

item_vocab  = {item: idx for idx, item in enumerate(unique_ids)}
inv_vocab   = {idx: item for item, idx in item_vocab.items()}

# Instantiate and load weights
_model      = GRU4Rec(vocab_size=vocab_size).to(DEVICE)
_model.load_state_dict(state_dict)
_model.eval()

# ─── INFERENCE ─────────────────────────────────────────────────────────────────
def recommend_next_items(session: list[str], top_k: int = 5) -> list[str]:
    idxs = [item_vocab[i] for i in session if i in item_vocab]
    if not idxs:
        return []
    tensor = torch.tensor([idxs], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = _model(tensor)
    last = logits[0, -1, :]
    topk = torch.topk(F.softmax(last, dim=-1), k=top_k).indices.tolist()
    return [inv_vocab[i] for i in topk]
