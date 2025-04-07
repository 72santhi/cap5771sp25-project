from flask import Flask, request, jsonify
import torch
import pandas as pd
import torch.nn as nn

# Load item mapping
data = pd.read_csv("session_events.csv")
item_vocab = {item: idx for idx, item in enumerate(set(data["Item_ID"]))}
inv_vocab = {idx: item for item, idx in item_vocab.items()}

# Define the GRU Model
class GRU4Rec(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.fc(out)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRU4Rec(len(item_vocab)).to(device)
model.load_state_dict(torch.load("session_rec_model.pth", map_location=device))
model.eval()

# Flask API
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        session = data.get("session", [])

        if not session:
            return jsonify({"error": "No session provided"}), 400

        session_ids = [item_vocab[item] for item in session if item in item_vocab]
        input_tensor = torch.tensor([session_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            last_step_logits = output[:, -1, :]
            top_items = torch.topk(last_step_logits, k=5).indices.squeeze().tolist()
            recommendations = [inv_vocab[idx] for idx in top_items]

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
