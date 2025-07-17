import numpy as np
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# 1) Load pretrained ResNet-18 with current weights API
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.eval()

# 2) Create a feature extractor by dropping the final classification layer
#    (ResNet children: conv1..layer4, avgpool, fc)
feature_net = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_net.eval()

# 3) Image preprocessing pipeline
_prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def embed_image(id_):
    """
    Embeds an image at data/images/{id_}.jpg (or .png/.jpeg).
    Returns a 512-d vector, or zeros if the file is missing or unreadable.
    """
    for ext in (".jpg", ".png", ".jpeg"):
        p = Path("data/images") / f"{id_}{ext}"
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                x = _prep(img).unsqueeze(0)  # shape (1,3,224,224)
                with torch.no_grad():
                    feat = feature_net(x).squeeze()  # shape (512,)
                return feat.cpu().numpy()
            except Exception:
                break

    # fallback if no image found or any error
    return np.zeros(512, dtype=float)

def add_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with an 'id' column and returns
    a new DataFrame with img_0 ... img_511 columns appended.
    """
    embeddings = np.stack([embed_image(i) for i in df["id"]])
    cols = [f"img_{j}" for j in range(embeddings.shape[1])]
    img_df = pd.DataFrame(embeddings, columns=cols)
    return pd.concat([df.reset_index(drop=True), img_df], axis=1)