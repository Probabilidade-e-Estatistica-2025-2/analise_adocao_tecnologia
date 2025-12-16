#!/usr/bin/env python3
"""
Script de Análise Crítica - Tipos de Variáveis
Demonstra os conceitos abordados nas aulas 4 e 5
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.analysis.variable_types import VariableClassifier, analise_completa_por_tipo

def main():
    # Carregar dados
    df = pd.read_csv(
        "src/data/database.csv",
        sep=";",
        decimal=","
    )
    
    print("\n" + "="*80)
    print("ANÁLISE CRÍTICA - CLASSIFICAÇÃO E ANÁLISE DE TIPOS DE VARIÁVEIS")
    print("Projeto: Adoção de Tecnologias | Probabilidade e Estatística 2025.2")
    print("="*80)
    
    # Mostrar dados brutos
    print("\n📊 DADOS CARREGADOS:")
    print(f"Total de registros: {len(df)}")
    print(f"Total de colunas: {len(df.columns)}")
    print(f"\nPrimeiras linhas:")
    print(df.head(10))
    
    # Análise completa por tipo
    analise_completa_por_tipo(df)
    
    # Gerar visualizações
    print("\n" + "="*80)
    print("GERANDO VISUALIZAÇÕES...")
    print("="*80)
    
    classifier = VariableClassifier()
    figs = classifier.gerar_visualizacoes(df)
    
    print(f"\n✓ {len(figs)} visualizações criadas com sucesso!")
    print("\nVisualização disponíveis:")
    for name in figs.keys():
        print(f"  • {name}")
    
    print("\n" + "="*80)
    print("✓ ANÁLISE COMPLETA FINALIZADA")
    print("="*80)
    
    return df, figs

if __name__ == "__main__":
    df, figs = main()
