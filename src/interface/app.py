import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

from src.analysis.stats import *
from src.analysis.variable_types import VariableClassifier

# Configuração visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

st.set_page_config(page_title="Adoção de Tecnologias", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(
        "src/data/database.csv",
        sep=";",
        decimal=","
    )

df = load_data()

# Filtros
st.sidebar.title("Filtros")

periodos = st.sidebar.multiselect(
    "Períodos",
    options=df["Periodo"].unique(),
    default=df["Periodo"].unique()
)

tecnologias = st.sidebar.multiselect(
    "Tecnologias",
    options=df["Tecnologia"].unique(),
    default=df["Tecnologia"].unique()
)

df_filtro = df[
    df["Periodo"].isin(periodos) &
    df["Tecnologia"].isin(tecnologias)
]

st.title("Dashboard Analítico – Adoção de Tecnologias")

# Métricas principais
c1, c2, c3, c4 = st.columns(4)

c1.metric("Taxa Média (%)", round(df_filtro["Taxa_Adocao_Percent"].mean(), 2))
c2.metric("Investimento Médio (Mi)", round(df_filtro["Investimento_Milhoes"].mean(), 2))
c3.metric("Satisfação Média", round(df_filtro["Satisfacao_Media"].mean(), 2))
c4.metric("Tempo Médio (meses)", round(df_filtro["Tempo_Implementacao_Meses"].mean(), 2))

# Estatística Descritiva
st.header("Estatística Descritiva")
st.dataframe(estatistica_descritiva(df_filtro), use_container_width=True)

# Evolução Temporal
st.header("Evolução Temporal")

tech = st.selectbox("Tecnologia", tecnologias)
df_time = df_filtro[df_filtro["Tecnologia"] == tech]

fig_evolucao = grafico_evolucao(df_time, tech)
fig_evolucao.set_size_inches(6, 3)
st.pyplot(fig_evolucao)

# Distribuição
st.header("Distribuição da Taxa de Adoção")

col1, col2 = st.columns(2)

with col1:
    fig_hist = histograma_adocao(df_filtro)
    fig_hist.set_size_inches(5, 3)
    st.pyplot(fig_hist)

with col2:
    fig_box = boxplot_adocao(df_filtro)
    fig_box.set_size_inches(5, 3)
    st.pyplot(fig_box)

# Comparações
st.header("Comparações entre Tecnologias")

metrica = st.selectbox("Métrica de comparação", NUMERIC_COLS)

fig_rank = ranking_medio(df_filtro, metrica)
fig_rank.set_size_inches(6, 3)
st.pyplot(fig_rank)

# Relações
st.header("Relações entre Variáveis")

col3, col4 = st.columns(2)

with col3:
    fig_disp = dispersao_investimento(df_filtro)
    fig_disp.set_size_inches(5, 3)
    st.pyplot(fig_disp)

with col4:
    fig_corr = matriz_correlacao(df_filtro)
    fig_corr.set_size_inches(5, 4)
    st.pyplot(fig_corr)

# Conclusões
st.header("Conclusões")
st.markdown("""
- A adoção das tecnologias apresenta crescimento consistente ao longo do tempo.
- Tecnologias com maior investimento e maior número de profissionais treinados tendem a apresentar maiores taxas de adoção.
- O tempo médio de implementação diminui à medida que a tecnologia amadurece.
- Cloud Computing e API REST destacam-se como líderes de mercado no período analisado.
""")

# ============================================================================
# ANÁLISE CRÍTICA - TIPOS DE VARIÁVEIS
# ============================================================================
st.header("🔍 ANÁLISE CRÍTICA: TIPOS DE VARIÁVEIS")

st.markdown("""
Análise estruturada conforme **Aulas 4 e 5 - Tipos de Variáveis**:
- **Qualitativa Nominal**: Sem ordem (Período, Tecnologia)
- **Qualitativa Ordinal**: Com ordem (Satisfação 1-10)
- **Quantitativa Discreta**: Valores inteiros (Empresas, Profissionais, Meses)
- **Quantitativa Contínua**: Valores reais (Taxa %, Investimento)
""")

# ============================================================================
# TAB 1: VARIÁVEIS NOMINAIS
# ============================================================================
with st.expander("📊 QUALITATIVA NOMINAL - Período e Tecnologia", expanded=True):
    st.markdown("""
    **Definição:** Categorias **sem ordem inerente**  
    **Análises:** Frequência, Moda, Tabelas Cruzadas  
    **Gráficos:** Barras, Setores  
    **❌ Evitar:** Média, Desvio Padrão, Correlação
    """)
    
    col_nom1, col_nom2 = st.columns(2)
    
    with col_nom1:
        st.subheader("Período (Nominal)")
        freq_periodo = df_filtro["Periodo"].value_counts().sort_index()
        freq_rel_periodo = (freq_periodo / len(df_filtro) * 100).round(1)
        
        fig_periodo, ax_periodo = plt.subplots(figsize=(8, 5))
        bars = ax_periodo.bar(freq_periodo.index, freq_periodo.values, color="steelblue", edgecolor="black")
        ax_periodo.set_title("Distribuição de Períodos (Frequência Absoluta)", fontsize=12, fontweight="bold")
        ax_periodo.set_ylabel("Frequência Absoluta")
        ax_periodo.set_xlabel("Período")
        plt.xticks(rotation=45)
        
        # Adicionar valores nas barras
        for i, (bar, val) in enumerate(zip(bars, freq_periodo.values)):
            ax_periodo.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'{val}\n({freq_rel_periodo.values[i]}%)', 
                          ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_periodo)
        
        # Estatísticas nominais
        st.write("**Estatísticas:**")
        st.write(f"- Moda (mais frequente): **{freq_periodo.idxmax()}** ({freq_periodo.max()} registros)")
        st.write(f"- Número de categorias: **{freq_periodo.nunique()}**")
        st.dataframe(pd.DataFrame({"Período": freq_periodo.index, "Frequência": freq_periodo.values, "Percentual": freq_rel_periodo.values}))
    
    with col_nom2:
        st.subheader("Tecnologia (Nominal)")
        freq_tech = df_filtro["Tecnologia"].value_counts()
        freq_rel_tech = (freq_tech / len(df_filtro) * 100).round(1)
        
        fig_tech, ax_tech = plt.subplots(figsize=(8, 5))
        bars = ax_tech.barh(freq_tech.index, freq_tech.values, color="coral", edgecolor="black")
        ax_tech.set_title("Distribuição de Tecnologias (Frequência Absoluta)", fontsize=12, fontweight="bold")
        ax_tech.set_xlabel("Frequência Absoluta")
        
        # Adicionar valores nas barras
        for i, (bar, val) in enumerate(zip(bars, freq_tech.values)):
            ax_tech.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val} ({freq_rel_tech.values[i]}%)',
                        va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_tech)
        
        st.write("**Estatísticas:**")
        st.write(f"- Moda (mais frequente): **{freq_tech.idxmax()}** ({freq_tech.max()} registros)")
        st.write(f"- Número de categorias: **{freq_tech.nunique()}**")
        st.dataframe(pd.DataFrame({"Tecnologia": freq_tech.index, "Frequência": freq_tech.values, "Percentual": freq_rel_tech.values}))
    
    # Tabela cruzada
    st.subheader("Tabela Cruzada: Período × Tecnologia")
    tabela_cruzada = pd.crosstab(df_filtro["Periodo"], df_filtro["Tecnologia"], margins=True)
    st.dataframe(tabela_cruzada)

# ============================================================================
# TAB 2: VARIÁVEIS ORDINAIS
# ============================================================================
with st.expander("📈 QUALITATIVA ORDINAL - Satisfação Média", expanded=True):
    st.markdown("""
    **Definição:** Categorias **com ordem natural**  
    **Análises:** Mediana, Moda, Quartis (NÃO média!)  
    **Gráficos:** Boxplot, Histograma, Distribuição  
    **Teste:** Mann-Whitney, Kruskal-Wallis
    """)
    
    col_ord1, col_ord2, col_ord3 = st.columns(3)
    
    with col_ord1:
        st.write("**Histograma da Satisfação**")
        fig_hist_ord, ax_hist_ord = plt.subplots(figsize=(8, 5))
        ax_hist_ord.hist(df_filtro["Satisfacao_Media"], bins=20, color="lightgreen", edgecolor="black", alpha=0.7)
        ax_hist_ord.axvline(df_filtro["Satisfacao_Media"].median(), color="red", linestyle="--", linewidth=2, label=f"Mediana: {df_filtro['Satisfacao_Media'].median():.2f}")
        ax_hist_ord.axvline(df_filtro["Satisfacao_Media"].mean(), color="blue", linestyle="--", linewidth=2, label=f"Média: {df_filtro['Satisfacao_Media'].mean():.2f}")
        ax_hist_ord.set_title("Distribuição da Satisfação", fontweight="bold")
        ax_hist_ord.set_xlabel("Satisfação (1-10)")
        ax_hist_ord.set_ylabel("Frequência")
        ax_hist_ord.legend()
        plt.tight_layout()
        st.pyplot(fig_hist_ord)
    
    with col_ord2:
        st.write("**Boxplot da Satisfação**")
        fig_box_ord, ax_box_ord = plt.subplots(figsize=(8, 5))
        bp = ax_box_ord.boxplot(df_filtro["Satisfacao_Media"], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        ax_box_ord.set_title("Boxplot - Satisfação", fontweight="bold")
        ax_box_ord.set_ylabel("Satisfação (1-10)")
        ax_box_ord.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_box_ord)
    
    with col_ord3:
        st.write("**Estatísticas Ordinais**")
        stats_ord = {
            "Mínimo": df_filtro["Satisfacao_Media"].min(),
            "Q1 (25%)": df_filtro["Satisfacao_Media"].quantile(0.25),
            "Mediana (50%)": df_filtro["Satisfacao_Media"].median(),
            "Moda": df_filtro["Satisfacao_Media"].mode().values[0] if len(df_filtro["Satisfacao_Media"].mode()) > 0 else "N/A",
            "Média": df_filtro["Satisfacao_Media"].mean(),
            "Q3 (75%)": df_filtro["Satisfacao_Media"].quantile(0.75),
            "Máximo": df_filtro["Satisfacao_Media"].max(),
            "IQR": df_filtro["Satisfacao_Media"].quantile(0.75) - df_filtro["Satisfacao_Media"].quantile(0.25),
            "Desvio Padrão": df_filtro["Satisfacao_Media"].std(),
        }
        st.dataframe(pd.DataFrame(stats_ord, index=["Valor"]).T)
    
    # Boxplot por tecnologia
    st.subheader("Boxplot: Satisfação por Tecnologia")
    fig_box_tech, ax_box_tech = plt.subplots(figsize=(12, 5))
    df_filtro.boxplot(column="Satisfacao_Media", by="Tecnologia", ax=ax_box_tech, patch_artist=True)
    plt.suptitle("")
    ax_box_tech.set_title("Satisfação por Tecnologia", fontweight="bold")
    ax_box_tech.set_ylabel("Satisfação (1-10)")
    ax_box_tech.set_xlabel("Tecnologia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_box_tech)

# ============================================================================
# TAB 3: VARIÁVEIS DISCRETAS
# ============================================================================
with st.expander("🔢 QUANTITATIVA DISCRETA - Contáveis/Inteiras", expanded=True):
    st.markdown("""
    **Definição:** Valores **inteiros contáveis**  
    **Análises:** Soma, Contagem, Média, Variância, Desvio Padrão  
    **Gráficos:** Histograma, Scatter, Boxplot  
    **Teste:** Pearson, t-test, ANOVA
    """)
    
    var_disc = st.selectbox("Selecione variável discreta:", 
                            ["Empresas_Adotantes", "Profissionais_Treinados", "Tempo_Implementacao_Meses"],
                            key="disc_select")
    
    col_disc1, col_disc2 = st.columns(2)
    
    with col_disc1:
        st.write(f"**Histograma - {var_disc}**")
        fig_hist_disc, ax_hist_disc = plt.subplots(figsize=(8, 5))
        ax_hist_disc.hist(df_filtro[var_disc], bins=20, color="mediumpurple", edgecolor="black", alpha=0.7)
        ax_hist_disc.axvline(df_filtro[var_disc].mean(), color="red", linestyle="--", linewidth=2, label=f"Média: {df_filtro[var_disc].mean():.2f}")
        ax_hist_disc.axvline(df_filtro[var_disc].median(), color="orange", linestyle="--", linewidth=2, label=f"Mediana: {df_filtro[var_disc].median():.2f}")
        ax_hist_disc.set_title(f"Distribuição de {var_disc}", fontweight="bold")
        ax_hist_disc.set_xlabel(var_disc)
        ax_hist_disc.set_ylabel("Frequência")
        ax_hist_disc.legend()
        plt.tight_layout()
        st.pyplot(fig_hist_disc)
    
    with col_disc2:
        st.write(f"**Boxplot - {var_disc}**")
        fig_box_disc, ax_box_disc = plt.subplots(figsize=(8, 5))
        bp = ax_box_disc.boxplot(df_filtro[var_disc], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('mediumpurple')
        ax_box_disc.set_title(f"Boxplot - {var_disc}", fontweight="bold")
        ax_box_disc.set_ylabel(var_disc)
        ax_box_disc.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_box_disc)
    
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        st.write("**Estatísticas Descritivas**")
        stats_disc = {
            "Soma": df_filtro[var_disc].sum(),
            "Contagem": len(df_filtro[var_disc]),
            "Média": df_filtro[var_disc].mean(),
            "Mediana": df_filtro[var_disc].median(),
            "Moda": df_filtro[var_disc].mode().values[0] if len(df_filtro[var_disc].mode()) > 0 else "N/A",
            "Mínimo": df_filtro[var_disc].min(),
            "Máximo": df_filtro[var_disc].max(),
            "Amplitude": df_filtro[var_disc].max() - df_filtro[var_disc].min(),
        }
        st.dataframe(pd.DataFrame(stats_disc, index=["Valor"]).T)
    
    with col_stats2:
        st.write("**Medidas de Dispersão**")
        stats_disp = {
            "Variância": df_filtro[var_disc].var(),
            "Desvio Padrão": df_filtro[var_disc].std(),
            "Coef. Variação (%)": (df_filtro[var_disc].std() / df_filtro[var_disc].mean()) * 100,
            "Q1 (25%)": df_filtro[var_disc].quantile(0.25),
            "Q2 (50%)": df_filtro[var_disc].quantile(0.50),
            "Q3 (75%)": df_filtro[var_disc].quantile(0.75),
            "IQR": df_filtro[var_disc].quantile(0.75) - df_filtro[var_disc].quantile(0.25),
        }
        st.dataframe(pd.DataFrame(stats_disp, index=["Valor"]).T)
    
    # Boxplot por tecnologia
    st.subheader(f"Boxplot: {var_disc} por Tecnologia")
    fig_box_disc_tech, ax_box_disc_tech = plt.subplots(figsize=(12, 5))
    df_filtro.boxplot(column=var_disc, by="Tecnologia", ax=ax_box_disc_tech, patch_artist=True)
    plt.suptitle("")
    ax_box_disc_tech.set_title(f"{var_disc} por Tecnologia", fontweight="bold")
    ax_box_disc_tech.set_ylabel(var_disc)
    ax_box_disc_tech.set_xlabel("Tecnologia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_box_disc_tech)

# ============================================================================
# TAB 4: VARIÁVEIS CONTÍNUAS
# ============================================================================
with st.expander("📊 QUANTITATIVA CONTÍNUA - Valores Reais", expanded=True):
    st.markdown("""
    **Definição:** Valores **reais em intervalo contínuo**  
    **Análises:** Média, Desvio Padrão, Distribuição, Correlação  
    **Gráficos:** Histograma, Scatter, Densidade  
    **Teste:** Pearson, t-test, Kolmogorov-Smirnov
    """)
    
    var_cont = st.selectbox("Selecione variável contínua:", 
                            ["Taxa_Adocao_Percent", "Investimento_Milhoes"],
                            key="cont_select")
    
    col_cont1, col_cont2 = st.columns(2)
    
    with col_cont1:
        st.write(f"**Histograma - {var_cont}**")
        fig_hist_cont, ax_hist_cont = plt.subplots(figsize=(8, 5))
        ax_hist_cont.hist(df_filtro[var_cont], bins=25, color="skyblue", edgecolor="black", alpha=0.7)
        ax_hist_cont.axvline(df_filtro[var_cont].mean(), color="red", linestyle="--", linewidth=2, label=f"Média: {df_filtro[var_cont].mean():.2f}")
        ax_hist_cont.axvline(df_filtro[var_cont].median(), color="orange", linestyle="--", linewidth=2, label=f"Mediana: {df_filtro[var_cont].median():.2f}")
        ax_hist_cont.set_title(f"Distribuição de {var_cont}", fontweight="bold")
        ax_hist_cont.set_xlabel(var_cont)
        ax_hist_cont.set_ylabel("Frequência")
        ax_hist_cont.legend()
        plt.tight_layout()
        st.pyplot(fig_hist_cont)
    
    with col_cont2:
        st.write(f"**Boxplot - {var_cont}**")
        fig_box_cont, ax_box_cont = plt.subplots(figsize=(8, 5))
        bp = ax_box_cont.boxplot(df_filtro[var_cont], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        ax_box_cont.set_title(f"Boxplot - {var_cont}", fontweight="bold")
        ax_box_cont.set_ylabel(var_cont)
        ax_box_cont.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_box_cont)
    
    col_stats_c1, col_stats_c2 = st.columns(2)
    
    with col_stats_c1:
        st.write("**Estatísticas Centrais**")
        stats_cent = {
            "Média": df_filtro[var_cont].mean(),
            "Mediana": df_filtro[var_cont].median(),
            "Moda": df_filtro[var_cont].mode().values[0] if len(df_filtro[var_cont].mode()) > 0 else "N/A",
            "Mínimo": df_filtro[var_cont].min(),
            "Máximo": df_filtro[var_cont].max(),
            "Amplitude": df_filtro[var_cont].max() - df_filtro[var_cont].min(),
        }
        st.dataframe(pd.DataFrame(stats_cent, index=["Valor"]).T)
    
    with col_stats_c2:
        st.write("**Medidas de Dispersão e Forma**")
        stats_form = {
            "Variância": df_filtro[var_cont].var(),
            "Desvio Padrão": df_filtro[var_cont].std(),
            "Coef. Variação (%)": (df_filtro[var_cont].std() / df_filtro[var_cont].mean()) * 100,
            "Assimetria (Skewness)": df_filtro[var_cont].skew(),
            "Curtose (Kurtosis)": df_filtro[var_cont].kurtosis(),
            "Q1 (25%)": df_filtro[var_cont].quantile(0.25),
            "Q3 (75%)": df_filtro[var_cont].quantile(0.75),
            "IQR": df_filtro[var_cont].quantile(0.75) - df_filtro[var_cont].quantile(0.25),
        }
        st.dataframe(pd.DataFrame(stats_form, index=["Valor"]).T)
    
    # Boxplot por tecnologia
    st.subheader(f"Boxplot: {var_cont} por Tecnologia")
    fig_box_cont_tech, ax_box_cont_tech = plt.subplots(figsize=(12, 5))
    df_filtro.boxplot(column=var_cont, by="Tecnologia", ax=ax_box_cont_tech, patch_artist=True)
    plt.suptitle("")
    ax_box_cont_tech.set_title(f"{var_cont} por Tecnologia", fontweight="bold")
    ax_box_cont_tech.set_ylabel(var_cont)
    ax_box_cont_tech.set_xlabel("Tecnologia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_box_cont_tech)

# ============================================================================
# CORRELAÇÕES
# ============================================================================
with st.expander("🔗 ANÁLISE DE CORRELAÇÃO - Variáveis Contínuas", expanded=False):
    st.markdown("""
    **⚠️ IMPORTANTE:** Correlação **apenas para variáveis quantitativas contínuas**!
    Período e Tecnologia (nominais) foram **excluídas** desta análise.
    """)
    
    numeric_cols_for_corr = [
        "Empresas_Adotantes",
        "Taxa_Adocao_Percent",
        "Investimento_Milhoes",
        "Profissionais_Treinados",
        "Satisfacao_Media",
        "Tempo_Implementacao_Meses",
    ]
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        st.write("**Matriz de Correlação (Heatmap)**")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr_matrix = df_filtro[numeric_cols_for_corr].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                   square=True, ax=ax_corr, cbar_kws={"label": "Correlação"})
        ax_corr.set_title("Matriz de Correlação de Pearson", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr)
    
    with col_corr2:
        st.write("**Correlações Mais Fortes**")
        
        # Encontrar correlações mais fortes
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append({
                    "Var1": corr_matrix.columns[i],
                    "Var2": corr_matrix.columns[j],
                    "Correlação": corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_values).sort_values("Correlação", key=abs, ascending=False)
        st.dataframe(corr_df.head(10))

# ============================================================================
# COMPARAÇÕES ENTRE TECNOLOGIAS
# ============================================================================
st.header("⚖️ COMPARAÇÕES ENTRE TECNOLOGIAS")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.write("**Ranking Médio - Taxa de Adoção (%)**")
    fig_rank_taxa, ax_rank_taxa = plt.subplots(figsize=(8, 5))
    df_filtro.groupby("Tecnologia")["Taxa_Adocao_Percent"].mean().sort_values().plot(kind="barh", ax=ax_rank_taxa, color="steelblue")
    ax_rank_taxa.set_title("Taxa de Adoção Média por Tecnologia", fontweight="bold")
    ax_rank_taxa.set_xlabel("Taxa de Adoção (%)")
    plt.tight_layout()
    st.pyplot(fig_rank_taxa)

with col_comp2:
    st.write("**Ranking Médio - Investimento (Mi)**")
    fig_rank_inv, ax_rank_inv = plt.subplots(figsize=(8, 5))
    df_filtro.groupby("Tecnologia")["Investimento_Milhoes"].mean().sort_values().plot(kind="barh", ax=ax_rank_inv, color="coral")
    ax_rank_inv.set_title("Investimento Médio por Tecnologia", fontweight="bold")
    ax_rank_inv.set_xlabel("Investimento (Milhões R$)")
    plt.tight_layout()
    st.pyplot(fig_rank_inv)

# ============================================================================
# SCATTER PLOTS - RELAÇÕES
# ============================================================================
st.header("📈 RELAÇÕES ENTRE VARIÁVEIS CONTÍNUAS")

col_scatter1, col_scatter2 = st.columns(2)

with col_scatter1:
    st.write("**Investimento vs Taxa de Adoção**")
    fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 5))
    scatter1 = ax_scatter1.scatter(df_filtro["Investimento_Milhoes"], df_filtro["Taxa_Adocao_Percent"], 
                                  alpha=0.6, c=df_filtro.index, cmap="viridis", s=100, edgecolors="black")
    
    # Regressão linear
    z = np.polyfit(df_filtro["Investimento_Milhoes"], df_filtro["Taxa_Adocao_Percent"], 1)
    p = np.poly1d(z)
    ax_scatter1.plot(df_filtro["Investimento_Milhoes"], p(df_filtro["Investimento_Milhoes"]), "r--", linewidth=2, label="Tendência")
    
    corr_inv_taxa = df_filtro["Investimento_Milhoes"].corr(df_filtro["Taxa_Adocao_Percent"])
    ax_scatter1.set_title(f"Investimento vs Taxa (Correlação: {corr_inv_taxa:.3f})", fontweight="bold")
    ax_scatter1.set_xlabel("Investimento (Milhões R$)")
    ax_scatter1.set_ylabel("Taxa de Adoção (%)")
    ax_scatter1.legend()
    ax_scatter1.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_scatter1)

with col_scatter2:
    st.write("**Profissionais Treinados vs Taxa de Adoção**")
    fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 5))
    scatter2 = ax_scatter2.scatter(df_filtro["Profissionais_Treinados"], df_filtro["Taxa_Adocao_Percent"],
                                  alpha=0.6, c=df_filtro.index, cmap="plasma", s=100, edgecolors="black")
    
    # Regressão linear
    z2 = np.polyfit(df_filtro["Profissionais_Treinados"], df_filtro["Taxa_Adocao_Percent"], 1)
    p2 = np.poly1d(z2)
    ax_scatter2.plot(df_filtro["Profissionais_Treinados"], p2(df_filtro["Profissionais_Treinados"]), "r--", linewidth=2, label="Tendência")
    
    corr_prof_taxa = df_filtro["Profissionais_Treinados"].corr(df_filtro["Taxa_Adocao_Percent"])
    ax_scatter2.set_title(f"Profissionais vs Taxa (Correlação: {corr_prof_taxa:.3f})", fontweight="bold")
    ax_scatter2.set_xlabel("Profissionais Treinados")
    ax_scatter2.set_ylabel("Taxa de Adoção (%)")
    ax_scatter2.legend()
    ax_scatter2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_scatter2)

# ============================================================================
# TABELA DE RESUMO ESTATÍSTICO
# ============================================================================
st.header("📋 RESUMO ESTATÍSTICO COMPLETO")
resumo_stats = estatistica_descritiva(df_filtro)
st.dataframe(resumo_stats, use_container_width=True)
