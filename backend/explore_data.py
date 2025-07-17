import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("data/train.csv")

# 2. Quick look
print("Shape:", df.shape)
print(df.info(), "\n")
print(df.describe(), "\n")
print("First 5 rows:\n", df.head(), "\n")

# 3. Visualize distribution of target
plt.figure()
sns.histplot(df["price"], kde=True)
plt.title("Price distribution")
plt.show()

# 4. Correlation heatmap (numeric features only)
plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# 5. Check for nulls
null_counts = df.isnull().sum().sort_values(ascending=False)
print("Missing values per column:\n", null_counts[null_counts > 0], "\n")