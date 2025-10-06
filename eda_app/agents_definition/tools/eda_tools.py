from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import sys



@tool("execute_python_code")
def execute_python_code(csv_filename: str, python_code: str) -> str:
    """
    Executa código Python para analisar dados do CSV.
    Use esta ferramenta quando precisar fazer análises customizadas que não são cobertas por analyze_csv_data.
    
    Args:
        csv_filename: Nome do arquivo CSV no diretório data/ (ex: 'dataset.csv')
        python_code: Código Python a ser executado. O DataFrame estará disponível como 'df'.
                    Exemplo: "print(df['coluna'].value_counts())"
    
    Returns:
        Resultado da execução em formato JSON string
    """
    try:
        # Construir o caminho completo do arquivo
        data_dir = Path("data")
        file_path = data_dir / csv_filename
        
        # Verificar se o arquivo existe
        if not file_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Arquivo {csv_filename} não encontrado no diretório data/"
            })
        
        # Criar código completo com importações e carregamento do CSV
        full_code = f"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# Carregar o CSV
file_path = Path("data") / "{csv_filename}"
df = pd.read_csv(file_path)

# Código do usuário
{python_code}
        """
        
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name
        
        try:
            # Executar código
            proc = subprocess.Popen(
                [sys.executable, tmp_path],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(timeout=30)
            exit_code = proc.returncode
            
            # Limpar arquivo temporário
            os.remove(tmp_path)
            
            if exit_code == 0:
                return json.dumps({
                    "success": True,
                    "output": stdout,
                    "code_executed": python_code
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": stderr,
                    "output": stdout
                })
                
        except subprocess.TimeoutExpired:
            proc.kill()
            os.remove(tmp_path)
            return json.dumps({
                "success": False,
                "error": "Código demorou muito para executar (timeout de 30s)"
            })
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Erro ao executar código: {str(e)}"
        })


@tool("describe_data")
def describe_data(csv_path: str) -> str:
    """Descreve os dados em um arquivo CSV, incluindo tipos de dados, estatísticas e valores ausentes."""
    try:
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:  # Se conseguiu separar as colunas corretamente
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(csv_path)  # Fallback para separador padrão
        
        description = "# Descrição dos Dados\n\n"
        description += f"**Dimensões:** {df.shape[0]} linhas e {df.shape[1]} colunas\n\n"
        description += f"**Tipos de Dados:**\n```\n{df.dtypes.to_string()}\n```\n\n"
        description += f"**Estatísticas Descritivas:**\n```\n{df.describe().to_string()}\n```\n\n"
        description += f"**Valores Ausentes:**\n```\n{df.isnull().sum().to_string()}\n```\n\n"
        description += f"**Primeiras 5 linhas:**\n```\n{df.head().to_string()}\n```"
        
        return description
    except Exception as e:
        return f"Erro ao analisar o arquivo: {str(e)}"

@tool("plot_distributions")
def plot_distributions(csv_path: str) -> str:
    """Plota e salva histogramas para variáveis numéricas e gráficos de contagem para variáveis categóricas."""
    try:
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(csv_path)
        
        charts_dir = 'charts'
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        plt.style.use('default')
        
        # Criar relatório markdown com gráficos incorporados
        markdown_report = "## 📊 Análise de Distribuições\n\n"
        
        for col in df.columns:
            plt.figure(figsize=(12, 6))
            
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Histograma de {col}')
                
                plt.subplot(1, 2, 2)
                sns.boxplot(y=df[col].dropna())
                plt.title(f'Boxplot de {col}')
                
                # Análise estatística
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                markdown_report += f"### {col} (Variável Numérica)\n\n"
                markdown_report += f"![Distribuição de {col}](charts/{col}_distribution.png)\n\n"
                markdown_report += f"**Análise Estatística:**\n"
                markdown_report += f"- **Média:** {mean_val:.2f}\n"
                markdown_report += f"- **Mediana:** {median_val:.2f}\n"
                markdown_report += f"- **Desvio Padrão:** {std_val:.2f}\n"
                
                # Interpretação da distribuição
                if abs(mean_val - median_val) < 0.1 * std_val:
                    markdown_report += f"- **Distribuição:** Aproximadamente simétrica\n"
                elif mean_val > median_val:
                    markdown_report += f"- **Distribuição:** Assimétrica à direita (cauda longa à direita)\n"
                else:
                    markdown_report += f"- **Distribuição:** Assimétrica à esquerda (cauda longa à esquerda)\n"
                
            else:
                value_counts = df[col].value_counts().head(20)
                sns.countplot(y=df[col], order=value_counts.index)
                plt.title(f'Contagem de {col}')
                
                markdown_report += f"### {col} (Variável Categórica)\n\n"
                markdown_report += f"![Distribuição de {col}](charts/{col}_distribution.png)\n\n"
                markdown_report += f"**Análise Categórica:**\n"
                markdown_report += f"- **Valores únicos:** {df[col].nunique()}\n"
                markdown_report += f"- **Valor mais frequente:** {value_counts.index[0]} ({value_counts.iloc[0]} ocorrências)\n"
                markdown_report += f"- **Distribuição:** "
                
                # Análise da distribuição categórica
                if value_counts.iloc[0] > len(df) * 0.5:
                    markdown_report += "Altamente concentrada em uma categoria\n"
                elif value_counts.std() < value_counts.mean() * 0.5:
                    markdown_report += "Relativamente uniforme entre categorias\n"
                else:
                    markdown_report += "Distribuição variada entre categorias\n"
                
            chart_path = f"{charts_dir}/{col}_distribution.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            markdown_report += "\n---\n\n"
            
        return markdown_report
    except Exception as e:
        return f"Erro ao gerar gráficos de distribuição: {str(e)}"

@tool("plot_correlations")
def plot_correlations(csv_path: str) -> str:
    """Plota e salva um heatmap de correlação para variáveis numéricas."""
    try:
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(csv_path)
        
        charts_dir = 'charts'
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return "## 🔗 Análise de Correlações\n\nNão há variáveis numéricas suficientes para análise de correlação."
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Heatmap de Correlação')
        chart_path = f"{charts_dir}/correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Criar relatório markdown com análise de correlações
        markdown_report = "## 🔗 Análise de Correlações\n\n"
        markdown_report += "![Heatmap de Correlação](charts/correlation_heatmap.png)\n\n"
        
        # Encontrar correlações mais fortes
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.1:  # Apenas correlações significativas
                    correlation_pairs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Ordenar por valor absoluto da correlação
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        markdown_report += "### 📈 Principais Correlações Encontradas\n\n"
        
        if correlation_pairs:
            for pair in correlation_pairs[:10]:  # Top 10 correlações
                corr_strength = ""
                if abs(pair['correlation']) > 0.7:
                    corr_strength = "**Forte**"
                elif abs(pair['correlation']) > 0.4:
                    corr_strength = "**Moderada**"
                else:
                    corr_strength = "Fraca"
                
                corr_direction = "positiva" if pair['correlation'] > 0 else "negativa"
                
                markdown_report += f"- **{pair['var1']} ↔ {pair['var2']}**: {pair['correlation']:.3f} "
                markdown_report += f"({corr_strength} correlação {corr_direction})\n"
        else:
            markdown_report += "Não foram encontradas correlações significativas entre as variáveis.\n"
        
        markdown_report += "\n### 🔍 Interpretação das Correlações\n\n"
        markdown_report += "- **Correlação > 0.7**: Relação linear forte\n"
        markdown_report += "- **Correlação 0.4-0.7**: Relação linear moderada\n"
        markdown_report += "- **Correlação < 0.4**: Relação linear fraca\n"
        markdown_report += "- **Correlação positiva**: As variáveis tendem a aumentar juntas\n"
        markdown_report += "- **Correlação negativa**: Quando uma aumenta, a outra tende a diminuir\n\n"
        
        return markdown_report
    except Exception as e:
        return f"Erro ao gerar heatmap de correlação: {str(e)}"

@tool("detect_outliers")
def detect_outliers(csv_path: str) -> str:
    """Detecta e plota outliers usando boxplots para variáveis numéricas."""
    try:
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(csv_path)
        
        charts_dir = 'charts'
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return "## 🚨 Detecção de Outliers\n\nNão há variáveis numéricas para detecção de outliers."
        
        markdown_report = "## 🚨 Detecção de Outliers\n\n"
        total_outliers = 0
        
        for col in numeric_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Boxplot
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df[col].dropna())
            plt.title(f'Boxplot de {col}')
            
            # Histograma com outliers destacados
            plt.subplot(1, 2, 2)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            normal_values = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col]
            
            plt.hist(normal_values, bins=30, alpha=0.7, label='Valores normais', color='blue')
            plt.hist(outliers, bins=30, alpha=0.7, label='Outliers', color='red')
            plt.title(f'Distribuição de {col} com Outliers')
            plt.legend()
            
            chart_path = f"{charts_dir}/{col}_outliers.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Adicionar ao relatório markdown
            markdown_report += f"### {col}\n\n"
            markdown_report += f"![Análise de Outliers - {col}](charts/{col}_outliers.png)\n\n"
            
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            total_outliers += outlier_count
            
            markdown_report += f"**Estatísticas de Outliers:**\n"
            markdown_report += f"- **Quantidade:** {outlier_count} outliers detectados\n"
            markdown_report += f"- **Percentual:** {outlier_percentage:.1f}% dos dados\n"
            markdown_report += f"- **Limites:** [{lower_bound:.2f}, {upper_bound:.2f}]\n"
            
            # Interpretação
            if outlier_percentage > 10:
                markdown_report += f"- **Interpretação:** ⚠️ Alto número de outliers - investigar possíveis erros nos dados\n"
            elif outlier_percentage > 5:
                markdown_report += f"- **Interpretação:** ⚡ Número moderado de outliers - analisar se são valores legítimos\n"
            else:
                markdown_report += f"- **Interpretação:** ✅ Poucos outliers - distribuição relativamente normal\n"
            
            if outlier_count > 0:
                markdown_report += f"- **Valores extremos:** Min: {outliers.min():.2f}, Max: {outliers.max():.2f}\n"
            
            markdown_report += "\n---\n\n"
        
        # Resumo geral
        markdown_report += f"### 📊 Resumo Geral de Outliers\n\n"
        markdown_report += f"- **Total de outliers:** {total_outliers}\n"
        markdown_report += f"- **Variáveis analisadas:** {len(numeric_df.columns)}\n"
        markdown_report += f"- **Percentual geral:** {(total_outliers / (len(df) * len(numeric_df.columns))) * 100:.1f}% dos valores\n\n"
        
        markdown_report += "**Recomendações:**\n"
        markdown_report += "- Investigar a origem dos outliers (erros de medição, valores extremos legítimos, etc.)\n"
        markdown_report += "- Considerar transformações de dados se necessário\n"
        markdown_report += "- Avaliar o impacto dos outliers nas análises subsequentes\n\n"
        
        return markdown_report
    except Exception as e:
        return f"Erro ao detectar outliers: {str(e)}"

@tool("create_custom_chart")
def create_custom_chart(csv_path: str, chart_request: str) -> str:
    """Cria um gráfico personalizado baseado na solicitação do usuário."""
    try:
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(csv_path)
        
        charts_dir = 'charts'
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # Analisar a solicitação e gerar código Python
        chart_code = generate_chart_code(df, chart_request)
        
        # Executar o código gerado
        exec_globals = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'pd': pd,
            'np': np,
            'charts_dir': charts_dir
        }
        
        exec(chart_code, exec_globals)
        
        # Criar relatório markdown
        markdown_report = f"## 🎨 Gráfico Personalizado\n\n"
        markdown_report += f"**Solicitação:** {chart_request}\n\n"
        markdown_report += f"![Gráfico Personalizado](charts/custom_chart.png)\n\n"
        markdown_report += f"### 💻 Código Python Gerado\n\n"
        markdown_report += f"```python\n{chart_code}\n```\n\n"
        
        return markdown_report
        
    except Exception as e:
        return f"Erro ao criar gráfico personalizado: {str(e)}"

def generate_chart_code(df, request):
    """Gera código Python para criar gráficos baseado na solicitação do usuário."""
    
    # Analisar colunas disponíveis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    request_lower = request.lower()
    
    # Detectar tipo de gráfico solicitado
    if any(word in request_lower for word in ['scatter', 'dispersão', 'correlação entre']):
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            # Tentar identificar colunas específicas mencionadas na solicitação
            for col in numeric_cols:
                if col.lower() in request_lower:
                    if col1 == numeric_cols[0]:
                        col1 = col
                    else:
                        col2 = col
                        break
            
            code = f"""
plt.figure(figsize=(10, 6))
plt.scatter(df['{col1}'], df['{col2}'], alpha=0.6)
plt.xlabel('{col1}')
plt.ylabel('{col2}')
plt.title('Gráfico de Dispersão: {col1} vs {col2}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas numéricas suficientes para gráfico de dispersão"
            
    elif any(word in request_lower for word in ['bar', 'barra', 'contagem', 'frequência']):
        if categorical_cols:
            col = categorical_cols[0]
            # Tentar identificar coluna específica
            for c in categorical_cols:
                if c.lower() in request_lower:
                    col = c
                    break
            
            code = f"""
plt.figure(figsize=(12, 6))
value_counts = df['{col}'].value_counts().head(15)
plt.bar(range(len(value_counts)), value_counts.values)
plt.xlabel('{col}')
plt.ylabel('Frequência')
plt.title('Gráfico de Barras: {col}')
plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas categóricas disponíveis"
            
    elif any(word in request_lower for word in ['line', 'linha', 'temporal', 'tempo']):
        if numeric_cols:
            col = numeric_cols[0]
            # Tentar identificar coluna específica
            for c in numeric_cols:
                if c.lower() in request_lower:
                    col = c
                    break
            
            code = f"""
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['{col}'], marker='o', linewidth=2, markersize=4)
plt.xlabel('Índice')
plt.ylabel('{col}')
plt.title('Gráfico de Linha: {col}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas numéricas disponíveis"
            
    elif any(word in request_lower for word in ['pie', 'pizza', 'setores']):
        if categorical_cols:
            col = categorical_cols[0]
            # Tentar identificar coluna específica
            for c in categorical_cols:
                if c.lower() in request_lower:
                    col = c
                    break
            
            code = f"""
plt.figure(figsize=(10, 8))
value_counts = df['{col}'].value_counts().head(10)
plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gráfico de Pizza: {col}')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas categóricas disponíveis"
            
    elif any(word in request_lower for word in ['box', 'boxplot', 'caixa']):
        if numeric_cols:
            col = numeric_cols[0]
            # Tentar identificar coluna específica
            for c in numeric_cols:
                if c.lower() in request_lower:
                    col = c
                    break
            
            code = f"""
plt.figure(figsize=(8, 6))
plt.boxplot(df['{col}'].dropna())
plt.ylabel('{col}')
plt.title('Boxplot: {col}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas numéricas disponíveis"
            
    elif any(word in request_lower for word in ['hist', 'histograma', 'distribuição']):
        if numeric_cols:
            col = numeric_cols[0]
            # Tentar identificar coluna específica
            for c in numeric_cols:
                if c.lower() in request_lower:
                    col = c
                    break
            
            code = f"""
plt.figure(figsize=(10, 6))
plt.hist(df['{col}'].dropna(), bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('{col}')
plt.ylabel('Frequência')
plt.title('Histograma: {col}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Não há colunas numéricas disponíveis"
            
    else:
        # Gráfico padrão baseado nos dados disponíveis
        if numeric_cols and categorical_cols:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            code = f"""
plt.figure(figsize=(12, 6))
df_grouped = df.groupby('{cat_col}')['{num_col}'].mean().head(10)
plt.bar(range(len(df_grouped)), df_grouped.values)
plt.xlabel('{cat_col}')
plt.ylabel('Média de {num_col}')
plt.title('Gráfico Personalizado: Média de {num_col} por {cat_col}')
plt.xticks(range(len(df_grouped)), df_grouped.index, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        elif numeric_cols:
            col = numeric_cols[0]
            code = f"""
plt.figure(figsize=(10, 6))
plt.hist(df['{col}'].dropna(), bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('{col}')
plt.ylabel('Frequência')
plt.title('Histograma: {col}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{{charts_dir}}/custom_chart.png', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            code = "# Erro: Dados insuficientes para gerar gráfico"
    
    return code

def run_eda_analysis(file_path, question):
    """
    Função que executa a análise EDA usando create_eda_crew
    """
    from pathlib import Path
    from agents_definition.crews.eda_crew import create_eda_crew

    try:
        
        
        # Acionar o create_eda_crew
        result = create_eda_crew(str(file_path), question)
        
        return {
            "success": True,
            "result": result
            }
    
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro ao executar análise EDA: {str(e)}"
        }
