import pandas as pd
from ctgan import CTGAN
import argparse

def augment_tabular(src, dst, n_samples, epochs):
    df = pd.read_csv(src)
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    print(f"⚙️  Fitting CTGAN for {epochs} epochs on {len(df)} rows…")
    gan = CTGAN(epochs=epochs, batch_size=128, pac=1)
    gan.fit(df, discrete_columns=cat_cols)

    print(f"⚙️  Sampling {n_samples} synthetic rows…")
    synth = gan.sample(n_samples)
    synth = synth[df.columns]
    out = pd.concat([df, synth], ignore_index=True)
    out.to_csv(dst, index=False)
    print(f"✅  Augmented data written to {dst} ({len(out)} rows total)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",       default="data/train.csv")
    p.add_argument("--dst",       default="data/train_aug.csv")
    p.add_argument("--samples",   type=int, default=2000)
    p.add_argument("--epochs",    type=int, default=30)
    args = p.parse_args()

    augment_tabular(args.src, args.dst, args.samples, args.epochs)