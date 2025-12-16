"""
Módulo de Classificação e Análise de Tipos de Variáveis
=========================================================

CLASSIFICAÇÃO DE VARIÁVEIS:
- QUALITATIVA (Categórica): Expressa características/atributos
  * Nominal: Sem ordem (ex: Tecnologia, Cor, Cidade)
  * Ordinal: Com ordem natural (ex: Satisfação 1-10, Nível de Experiência)

- QUANTITATIVA (Numérica): Expressa quantidades/medidas
  * Discreta: Valores inteiros, contáveis (ex: Número de empresas)
  * Contínua: Qualquer valor no intervalo, mensuráveis (ex: Investimento em milhões)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class VariableClassifier:
    """Classifica e analisa tipos de variáveis do dataset"""
    
    VARIABLE_TYPES = {
        # QUALITATIVA NOMINAL (sem ordem)
        "Periodo": {"type": "QUALITATIVA_NOMINAL", "description": "Período temporal (sem ordem inerente)"},
        "Tecnologia": {"type": "QUALITATIVA_NOMINAL", "description": "Tipo de tecnologia adotada"},
        
        # QUALITATIVA ORDINAL (com ordem)
        "Satisfacao_Media": {"type": "QUALITATIVA_ORDINAL", "description": "Satisfação em escala 0-10 (ordinal)"},
        
        # QUANTITATIVA DISCRETA (valores inteiros)
        "Empresas_Adotantes": {"type": "QUANTITATIVA_DISCRETA", "description": "Número de empresas (contável)"},
        "Profissionais_Treinados": {"type": "QUANTITATIVA_DISCRETA", "description": "Quantidade de profissionais (contável)"},
        "Tempo_Implementacao_Meses": {"type": "QUANTITATIVA_DISCRETA", "description": "Meses de implementação (inteiros)"},
        
        # QUANTITATIVA CONTÍNUA (valores reais)
        "Taxa_Adocao_Percent": {"type": "QUANTITATIVA_CONTINUA", "description": "Percentual de adoção (0-100%)"},
        "Investimento_Milhoes": {"type": "QUANTITATIVA_CONTINUA", "description": "Investimento em milhões (valores reais)"},
    }
    
    @staticmethod
    def get_variable_classification(df):
        """Retorna classificação de todas as variáveis"""
        classification = {}
        for col in df.columns:
            if col in VariableClassifier.VARIABLE_TYPES:
                classification[col] = VariableClassifier.VARIABLE_TYPES[col]
            else:
                classification[col] = {"type": "DESCONHECIDO", "description": "Classificação não definida"}
        return classification

    @staticmethod
    def analise_nominal(df, variavel):
        """Análise de variável QUALITATIVA NOMINAL"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE: {variavel} (QUALITATIVA NOMINAL)")
        print(f"{'='*60}")
        
        freq = df[variavel].value_counts()
        freq_rel = df[variavel].value_counts(normalize=True) * 100
        
        resultado = {
            "frequencia": freq,
            "frequencia_relativa": freq_rel,
            "moda": freq.idxmax(),
            "numero_categorias": df[variavel].nunique()
        }
        
        print(f"Moda (mais frequente): {resultado['moda']}")
        print(f"Número de categorias: {resultado['numero_categorias']}")
        print(f"\nDistribuição:")
        print(freq_rel.round(2))
        
        return resultado

    @staticmethod
    def analise_ordinal(df, variavel):
        """Análise de variável QUALITATIVA ORDINAL"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE: {variavel} (QUALITATIVA ORDINAL)")
        print(f"{'='*60}")
        
        resultado = {
            "media": df[variavel].mean(),
            "mediana": df[variavel].median(),
            "moda": df[variavel].mode().values[0] if len(df[variavel].mode()) > 0 else None,
            "min": df[variavel].min(),
            "max": df[variavel].max(),
            "std": df[variavel].std(),
            "q1": df[variavel].quantile(0.25),
            "q3": df[variavel].quantile(0.75)
        }
        
        print(f"Mediana (posição central): {resultado['mediana']:.2f}")
        print(f"Média: {resultado['media']:.2f}")
        print(f"Moda (mais frequente): {resultado['moda']:.2f}")
        print(f"Amplitude: {resultado['min']:.2f} a {resultado['max']:.2f}")
        print(f"Desvio Padrão: {resultado['std']:.2f}")
        print(f"IQR (Q1-Q3): {resultado['q1']:.2f} - {resultado['q3']:.2f}")
        
        return resultado

    @staticmethod
    def analise_discreta(df, variavel):
        """Análise de variável QUANTITATIVA DISCRETA"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE: {variavel} (QUANTITATIVA DISCRETA)")
        print(f"{'='*60}")
        
        resultado = {
            "media": df[variavel].mean(),
            "mediana": df[variavel].median(),
            "moda": df[variavel].mode().values[0] if len(df[variavel].mode()) > 0 else None,
            "variancia": df[variavel].var(),
            "desvio_padrao": df[variavel].std(),
            "coef_variacao": (df[variavel].std() / df[variavel].mean()) * 100,
            "min": df[variavel].min(),
            "max": df[variavel].max(),
            "amplitude": df[variavel].max() - df[variavel].min(),
            "soma": df[variavel].sum()
        }
        
        print(f"Soma Total: {resultado['soma']}")
        print(f"Média: {resultado['media']:.2f}")
        print(f"Mediana: {resultado['mediana']:.2f}")
        print(f"Amplitude: {resultado['amplitude']} (de {resultado['min']} a {resultado['max']})")
        print(f"Variância: {resultado['variancia']:.2f}")
        print(f"Desvio Padrão: {resultado['desvio_padrao']:.2f}")
        print(f"Coeficiente de Variação: {resultado['coef_variacao']:.2f}%")
        
        return resultado

    @staticmethod
    def analise_continua(df, variavel):
        """Análise de variável QUANTITATIVA CONTÍNUA"""
        print(f"\n{'='*60}")
        print(f"ANÁLISE: {variavel} (QUANTITATIVA CONTÍNUA)")
        print(f"{'='*60}")
        
        resultado = {
            "media": df[variavel].mean(),
            "mediana": df[variavel].median(),
            "variancia": df[variavel].var(),
            "desvio_padrao": df[variavel].std(),
            "coef_variacao": (df[variavel].std() / df[variavel].mean()) * 100,
            "min": df[variavel].min(),
            "max": df[variavel].max(),
            "amplitude": df[variavel].max() - df[variavel].min(),
            "q1": df[variavel].quantile(0.25),
            "q2": df[variavel].quantile(0.50),
            "q3": df[variavel].quantile(0.75),
            "iqr": df[variavel].quantile(0.75) - df[variavel].quantile(0.25),
            "assimetria": df[variavel].skew(),
            "curtose": df[variavel].kurtosis()
        }
        
        print(f"Média: {resultado['media']:.4f}")
        print(f"Mediana: {resultado['mediana']:.4f}")
        print(f"Amplitude: {resultado['amplitude']:.4f} (de {resultado['min']:.4f} a {resultado['max']:.4f})")
        print(f"Variância: {resultado['variancia']:.4f}")
        print(f"Desvio Padrão: {resultado['desvio_padrao']:.4f}")
        print(f"Coeficiente de Variação: {resultado['coef_variacao']:.2f}%")
        print(f"IQR (Amplitude Interquartílica): {resultado['iqr']:.4f}")
        print(f"Assimetria (Skewness): {resultado['assimetria']:.4f}")
        print(f"Curtose (Kurtosis): {resultado['curtose']:.4f}")
        
        return resultado

    @staticmethod
    def gerar_visualizacoes(df):
        """Gera visualizações apropriadas para cada tipo de variável"""
        figs = {}
        
        # NOMINAL: Gráfico de Barras (Período)
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        df["Periodo"].value_counts().plot(kind="bar", ax=ax1, color="skyblue")
        ax1.set_title("Período - QUALITATIVA NOMINAL", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Frequência")
        plt.xticks(rotation=45)
        figs["periodo_nominal"] = fig1
        
        # NOMINAL: Gráfico de Barras (Tecnologia)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df["Tecnologia"].value_counts().plot(kind="bar", ax=ax2, color="lightcoral")
        ax2.set_title("Tecnologia - QUALITATIVA NOMINAL", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Frequência")
        plt.xticks(rotation=45)
        figs["tecnologia_nominal"] = fig2
        
        # ORDINAL: Histograma + Boxplot (Satisfação)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))
        ax3a.hist(df["Satisfacao_Media"], bins=15, color="lightgreen", edgecolor="black")
        ax3a.set_title("Satisfação - Histograma", fontweight="bold")
        ax3a.set_xlabel("Satisfação (1-10)")
        ax3b.boxplot(df["Satisfacao_Media"])
        ax3b.set_title("Satisfação - Boxplot", fontweight="bold")
        ax3b.set_ylabel("Satisfação (1-10)")
        figs["satisfacao_ordinal"] = fig3
        
        # DISCRETA: Scatter plot (Empresas vs Profissionais)
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.scatter(df["Empresas_Adotantes"], df["Profissionais_Treinados"], alpha=0.6, color="purple")
        ax4.set_title("Empresas vs Profissionais Treinados - QUANTITATIVA DISCRETA", fontweight="bold")
        ax4.set_xlabel("Empresas Adotantes (contáveis)")
        ax4.set_ylabel("Profissionais Treinados (contáveis)")
        figs["discreta_scatter"] = fig4
        
        # CONTÍNUA: Histograma (Taxa de Adoção)
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        ax5.hist(df["Taxa_Adocao_Percent"], bins=20, color="orange", edgecolor="black")
        ax5.set_title("Taxa de Adoção - QUANTITATIVA CONTÍNUA", fontweight="bold")
        ax5.set_xlabel("Taxa (%)")
        ax5.axvline(df["Taxa_Adocao_Percent"].mean(), color="red", linestyle="--", label=f"Média: {df['Taxa_Adocao_Percent'].mean():.2f}%")
        ax5.legend()
        figs["taxa_continua_hist"] = fig5
        
        # CONTÍNUA: Histograma (Investimento)
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        ax6.hist(df["Investimento_Milhoes"], bins=20, color="cyan", edgecolor="black")
        ax6.set_title("Investimento - QUANTITATIVA CONTÍNUA", fontweight="bold")
        ax6.set_xlabel("Investimento (Milhões R$)")
        ax6.axvline(df["Investimento_Milhoes"].mean(), color="red", linestyle="--", label=f"Média: {df['Investimento_Milhoes'].mean():.2f}M")
        ax6.legend()
        figs["investimento_continua_hist"] = fig6
        
        return figs

    @staticmethod
    def relatorio_critico(df):
        """Gera relatório crítico da análise"""
        print("\n" + "="*70)
        print("ANÁLISE CRÍTICA DO PROJETO - TIPOS DE VARIÁVEIS")
        print("="*70)
        
        print("\n📊 RESUMO DAS VARIÁVEIS:")
        print("-" * 70)
        
        for col, info in VariableClassifier.VARIABLE_TYPES.items():
            if col in df.columns:
                print(f"\n{col}")
                print(f"  • Tipo: {info['type']}")
                print(f"  • Descrição: {info['description']}")
                print(f"  • Observações: {len(df[col])} registros")
        
        print("\n" + "="*70)
        print("CONSIDERAÇÕES CRÍTICAS:")
        print("="*70)
        
        consideracoes = """
1. VARIÁVEIS NOMINAIS (Periodo, Tecnologia):
   ✓ Adequadas para análise de frequência e distribuição
   ✗ NÃO apropriadas para cálculo de média/desvio padrão
   → Usar: moda, frequência, gráficos de barras, testes qui-quadrado

2. VARIÁVEIS ORDINAIS (Satisfacao_Media):
   ✓ Possuem ordem, podem usar mediana e quartis
   ✗ Distâncias entre valores não são uniformes
   → Usar: mediana, moda, gráficos boxplot, testes não-paramétricos

3. VARIÁVEIS DISCRETAS (Empresas, Profissionais, Tempo):
   ✓ São números inteiros, contáveis
   ✓ Adequadas para estatísticas descritivas completas
   ✗ Não há valores infinitos entre observações
   → Usar: média, variância, soma, frequências

4. VARIÁVEIS CONTÍNUAS (Taxa_Adocao, Investimento):
   ✓ Podem assumir qualquer valor no intervalo
   ✓ Adequadas para distribuições de probabilidade (Normal, etc)
   → Usar: média, desvio padrão, assimetria, curtose, correlação

⚠️ PROBLEMAS ENCONTRADOS:
   • Satisfação sendo tratada como número real quando deveria ser ordinal
   • Falta de análises específicas por tipo de variável
   • Correlação calculada incluindo variáveis nominais
   • Tratamento uniforme de variáveis com naturezas diferentes
"""
        print(consideracoes)
        
        return consideracoes


def analise_completa_por_tipo(df):
    """Executa análise completa para cada tipo de variável"""
    classifier = VariableClassifier()
    
    print("\n" + "🔍"*35)
    print("ANÁLISE ESTRUTURADA POR TIPO DE VARIÁVEL")
    print("🔍"*35)
    
    # Nominais
    print("\n" + "█"*70)
    print("QUALITATIVA NOMINAL - Sem ordem inerente")
    print("█"*70)
    classifier.analise_nominal(df, "Periodo")
    classifier.analise_nominal(df, "Tecnologia")
    
    # Ordinais
    print("\n" + "█"*70)
    print("QUALITATIVA ORDINAL - Com ordem natural")
    print("█"*70)
    classifier.analise_ordinal(df, "Satisfacao_Media")
    
    # Discretas
    print("\n" + "█"*70)
    print("QUANTITATIVA DISCRETA - Valores inteiros contáveis")
    print("█"*70)
    classifier.analise_discreta(df, "Empresas_Adotantes")
    classifier.analise_discreta(df, "Profissionais_Treinados")
    classifier.analise_discreta(df, "Tempo_Implementacao_Meses")
    
    # Contínuas
    print("\n" + "█"*70)
    print("QUANTITATIVA CONTÍNUA - Valores reais mensuráveis")
    print("█"*70)
    classifier.analise_continua(df, "Taxa_Adocao_Percent")
    classifier.analise_continua(df, "Investimento_Milhoes")
    
    # Relatório crítico
    classifier.relatorio_critico(df)
