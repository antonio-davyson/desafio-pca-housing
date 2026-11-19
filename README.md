# Análise de Componentes Principais (PCA) — Dataset Housing

Projeto desenvolvido para aplicar técnicas de **redução de dimensionalidade com PCA** no conjunto de dados `housing.csv` (545 imóveis e 13 atributos), facilitando a visualização de padrões e avaliando a perda de informação pelo erro de reconstrução.

## Estrutura do Repositório

- `pca_housing.ipynb` — Notebook completo com exploração, pré-processamento, PCA e visualizações.
- `pca_housing.py` — Script Python equivalente para execução direta via terminal.
- `housing.csv` — Base de dados original utilizada na análise.
- `variancia_explicada.png` — Gráfico da variância explicada individual e acumulada por componente.
- `projecao_pca_2d.png` — Dispersão dos imóveis projetados em `PC1` e `PC2`.

## Etapas Realizadas

1. **Inspeção dos dados:** verificação das 545 linhas, 13 colunas e ausência de valores nulos.
2. **Pré-processamento:**
   - Conversão das variáveis binárias (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) de `yes`/`no` para `1`/`0`.
   - Codificação ordinal de `furnishingstatus` (`unfurnished`: 0, `semi-furnished`: 1, `furnished`: 2).
   - Padronização das 13 variáveis com `StandardScaler` para evitar viés de escala (como `price` e `area`).
3. **Aplicação do PCA e Erro de Reconstrução:**
   - **2 componentes principais:** explicam **36.44%** da variância total (`PC1`: 24.62%, `PC2`: 11.81%), com **MSE de reconstrução (escala padronizada) = 0.6356**.
   - **8 componentes principais:** atingem **80.65%** da variância acumulada (patamar >= 80%).
   - **10 componentes principais:** atingem **90.55%** da variância acumulada (patamar >= 90%), reduzindo o **MSE de reconstrução para 0.0945**.

## Principais Fatores Identificados

- **PC1 (Valor e Porte do Imóvel):** fortemente influenciado por `price, area, bathrooms, airconditioning`, separando imóveis maiores e de maior valor de mercado dos imóveis mais compactos.
- **PC2 (Configuração Interna vs. Infraestrutura Externa):** puxado principalmente por `stories, basement, guestroom, prefarea`, diferenciando casas com mais pavimentos/quartos de imóveis com porão ou localização em área preferencial.

## Como Executar

```bash
pip install -r requirements.txt
python pca_housing.py
```
