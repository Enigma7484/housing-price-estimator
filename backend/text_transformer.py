# backend/text_transformer.py
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model     = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()

def embed_text(texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state[:,0]  # CLS token
    return out.numpy()

def add_text_features(df):
    embs = embed_text(df["description"].tolist())
    cols = [f"text_{i}" for i in range(embs.shape[1])]
    txt_df = pd.DataFrame(embs, columns=cols)
    return pd.concat([df.reset_index(drop=True), txt_df], axis=1)