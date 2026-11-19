import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# 1. Carregar e inspecionar o dataset
df = pd.read_csv("housing.csv")
print("Dimensões:", df.shape)
print(df.info())
print(df.head())

# 2. Pré-processamento das variáveis categóricas e padronização
df_proc = df.copy()

binary_cols = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
for col in binary_cols:
    df_proc[col] = df_proc[col].map({"yes": 1, "no": 0})

furnishing_map = {
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2,
}
df_proc["furnishingstatus"] = df_proc["furnishingstatus"].map(furnishing_map)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_proc)

# 3. Análise da variância explicada para todos os componentes
pca_full = PCA()
pca_full.fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

n_80 = int(np.argmax(cum_var >= 0.80) + 1)
n_90 = int(np.argmax(cum_var >= 0.90) + 1)
print(f"Componentes para >= 80% da variância: {n_80} ({cum_var[n_80 - 1]:.2%})")
print(f"Componentes para >= 90% da variância: {n_90} ({cum_var[n_90 - 1]:.2%})")

# 4. Redução para 2 componentes (visualização) e cálculo do erro de reconstrução
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
X_rec_2 = pca_2.inverse_transform(X_pca_2)
mse_2 = mean_squared_error(X_scaled, X_rec_2)

# Redução preservando >= 90% da variância
pca_90 = PCA(n_components=n_90)
X_pca_90 = pca_90.fit_transform(X_scaled)
X_rec_90 = pca_90.inverse_transform(X_pca_90)
mse_90 = mean_squared_error(X_scaled, X_rec_90)

print(f"Variância explicada (2 componentes): {cum_var[1]:.2%}")
print(f"Erro de reconstrução (MSE padronizado - 2 comps): {mse_2:.4f}")
print(f"Erro de reconstrução (MSE padronizado - {n_90} comps): {mse_90:.4f}")

# 5. Interpretação dos pesos (loadings) de cada atributo
loadings = pd.DataFrame(
    pca_2.components_.T,
    index=df_proc.columns,
    columns=["PC1", "PC2"],
)
print("\nPesos (Loadings) nos dois primeiros componentes:")
print(loadings.round(3))

# 6. Visualizações
plt.figure(figsize=(8, 4.5))
plt.bar(
    range(1, len(pca_full.explained_variance_ratio_) + 1),
    pca_full.explained_variance_ratio_ * 100,
    alpha=0.55,
    color="#6366f1",
    label="Individual",
)
plt.plot(
    range(1, len(cum_var) + 1),
    cum_var * 100,
    marker="o",
    color="#1e1b4b",
    label="Acumulada",
)
plt.axhline(80, color="#059669", linestyle="--", linewidth=1, label="80%")
plt.axhline(90, color="#dc2626", linestyle="--", linewidth=1, label="90%")
plt.title("Variância Explicada por Componente Principal")
plt.xlabel("Componente Principal")
plt.ylabel("Variância Explicada (%)")
plt.xticks(range(1, len(cum_var) + 1))
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("variancia_explicada.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sc = plt.scatter(
    X_pca_2[:, 0],
    X_pca_2[:, 1],
    c=df["price"] / 1e6,
    cmap="viridis",
    alpha=0.8,
    s=30,
)
plt.colorbar(sc, label="Preço do Imóvel (Milhões)")
plt.title("Projeção dos Imóveis no Espaço PCA (PC1 vs PC2)")
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]:.1%})")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("projecao_pca_2d.png", dpi=150)
plt.show()
