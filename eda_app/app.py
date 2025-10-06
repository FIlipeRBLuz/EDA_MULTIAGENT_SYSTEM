import os
import sys
import json
import asyncio
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional
import threading
import subprocess
import tempfile
from PIL import Image
import glob


# IMPORTANTE: Importar da biblioteca openai-agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agents import Agent, Runner, SQLiteSession, function_tool
except ImportError:
    import agents as openai_agents_lib
    Agent = openai_agents_lib.Agent
    Runner = openai_agents_lib.Runner
    SQLiteSession = openai_agents_lib.SQLiteSession
    function_tool = openai_agents_lib.function_tool

# Importar a função create_eda_crew
try:
    from agents_definition.crews.eda_crew import create_eda_crew
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("eda_crew", "agents/crews/eda_crew.py")
    eda_crew_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda_crew_module)
    create_eda_crew = eda_crew_module.create_eda_crew

# Configurar API Key
gen = os.getenv("OPENAI_KEY_API")
if gen is None:
    try:
        gen = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    except:
        gen = os.getenv("OPENAI_API_KEY")

#para casos do litellm
if gen:
    os.environ["OPENAI_API_KEY"] = gen

# Variáveis globais
if "selected_csv" not in st.session_state:
    st.session_state.selected_csv = None

if "eda_running" not in st.session_state:
    st.session_state.eda_running = False

if "eda_markdown" not in st.session_state:
    st.session_state.eda_markdown = None

def detect_csv_separator(file_path):
    """Detecta automaticamente o separador de um arquivo CSV"""
    import csv
    
    # Lista de separadores comuns para testar
    separators = [',', ';', '\t', '|', ':', ' ']
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        # Ler as primeiras linhas para análise
        sample = file.read(1024)
        file.seek(0)
        
        # Usar o Sniffer do CSV para detectar o separador
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=',;\t|: ').delimiter
            return delimiter
        except:
            # Se o Sniffer falhar, testar manualmente
            first_line = file.readline()
            
            # Contar ocorrências de cada separador
            separator_counts = {}
            for sep in separators:
                separator_counts[sep] = first_line.count(sep)
            
            # Retornar o separador mais comum (que não seja espaço se houver outros)
            most_common = max(separator_counts.items(), key=lambda x: x[1])
            if most_common[1] > 0:
                return most_common[0]
            else:
                return ','  # Default para vírgula

def read_csv_robust(file_path):
    """Lê um arquivo CSV de forma robusta, detectando automaticamente o separador"""
    try:
        # Detectar o separador
        separator = detect_csv_separator(file_path)
        
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=separator, encoding=encoding)
                # Verificar se a leitura foi bem-sucedida (mais de 1 coluna)
                if len(df.columns) > 1:
                    return df
            except:
                continue
        
        # Se tudo falhar, tentar com parâmetros padrão
        df = pd.read_csv(file_path)
        return df
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {str(e)}")
        return None, None, None


@function_tool
def analyze_csv_data(csv_filename: str, analysis_type: str, column_name: str = None) -> str:
    """
    Analisa dados do CSV para responder perguntas simples sem necessidade de visualizações.
    Use esta ferramenta para perguntas sobre estatísticas descritivas, valores, contagens, etc.
    
    Args:
        csv_filename: Nome do arquivo CSV no diretório data/ (ex: 'dataset.csv')
        analysis_type: Tipo de análise desejada. Opções:
            - 'describe': Estatísticas descritivas de uma coluna ou dataset completo
            - 'mean': Média de uma coluna numérica
            - 'median': Mediana de uma coluna numérica
            - 'std': Desvio padrão de uma coluna numérica
            - 'min': Valor mínimo de uma coluna
            - 'max': Valor máximo de uma coluna
            - 'count': Contagem de valores/linhas
            - 'unique': Valores únicos de uma coluna
            - 'value_counts': Contagem de cada valor único
            - 'missing': Valores faltantes
            - 'dtypes': Tipos de dados das colunas
            - 'columns': Lista de colunas do dataset
            - 'shape': Dimensões do dataset (linhas x colunas)
            - 'head': Primeiras linhas do dataset
            - 'correlation': Correlação entre colunas numéricas
        column_name: Nome da coluna para análise (opcional, depende do analysis_type)
    
    Returns:
        Resultado da análise em formato JSON string
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
        
        # Ler o CSV
        try:
            df = read_csv_robust(file_path)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Erro ao ler o arquivo CSV: {str(e)}"
            })
        
        # Executar análise baseada no tipo
        result_data = {}
        
        if analysis_type == 'describe':
            if column_name and column_name in df.columns:
                result_data = df[column_name].describe().to_dict()
            else:
                result_data = df.describe().to_dict()
        
        elif analysis_type == 'mean':
            if column_name and column_name in df.columns:
                result_data = {"mean": float(df[column_name].mean())}
            else:
                result_data = df.select_dtypes(include=['number']).mean().to_dict()
        
        elif analysis_type == 'median':
            if column_name and column_name in df.columns:
                result_data = {"median": float(df[column_name].median())}
            else:
                result_data = df.select_dtypes(include=['number']).median().to_dict()
        
        elif analysis_type == 'std':
            if column_name and column_name in df.columns:
                result_data = {"std": float(df[column_name].std())}
            else:
                result_data = df.select_dtypes(include=['number']).std().to_dict()
        
        elif analysis_type == 'min':
            if column_name and column_name in df.columns:
                result_data = {"min": float(df[column_name].min())}
            else:
                result_data = df.select_dtypes(include=['number']).min().to_dict()
        
        elif analysis_type == 'max':
            if column_name and column_name in df.columns:
                result_data = {"max": float(df[column_name].max())}
            else:
                result_data = df.select_dtypes(include=['number']).max().to_dict()
        
        elif analysis_type == 'count':
            if column_name and column_name in df.columns:
                result_data = {"count": int(df[column_name].count())}
            else:
                result_data = {"total_rows": len(df)}
        
        elif analysis_type == 'unique':
            if column_name and column_name in df.columns:
                unique_values = df[column_name].unique().tolist()
                result_data = {
                    "unique_count": len(unique_values),
                    "unique_values": unique_values[:50]  # Limitar a 50 valores
                }
            else:
                result_data = {"error": "column_name é obrigatório para analysis_type='unique'"}
        
        elif analysis_type == 'value_counts':
            if column_name and column_name in df.columns:
                result_data = df[column_name].value_counts().head(20).to_dict()
            else:
                result_data = {"error": "column_name é obrigatório para analysis_type='value_counts'"}
        
        elif analysis_type == 'missing':
            if column_name and column_name in df.columns:
                result_data = {
                    "missing_count": int(df[column_name].isna().sum()),
                    "missing_percentage": float(df[column_name].isna().sum() / len(df) * 100)
                }
            else:
                missing_data = df.isna().sum()
                result_data = {col: int(count) for col, count in missing_data.items() if count > 0}
        
        elif analysis_type == 'dtypes':
            result_data = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        elif analysis_type == 'columns':
            result_data = {"columns": df.columns.tolist()}
        
        elif analysis_type == 'shape':
            result_data = {"rows": df.shape[0], "columns": df.shape[1]}
        
        elif analysis_type == 'head':
            result_data = df.head(10).to_dict(orient='records')
        
        elif analysis_type == 'correlation':
            corr_matrix = df.select_dtypes(include=['number']).corr()
            result_data = corr_matrix.to_dict()
        
        else:
            return json.dumps({
                "success": False,
                "error": f"Tipo de análise '{analysis_type}' não reconhecido"
            })
        
        return json.dumps({
            "success": True,
            "analysis_type": analysis_type,
            "column": column_name,
            "result": result_data,
            "file_info": {
                "filename": csv_filename,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Erro ao analisar dados: {str(e)}"
        })

@function_tool
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

@function_tool
def run_eda_analysis(csv_filename: str, question: str) -> str:
    """
    Executa uma análise exploratória de dados (EDA) completa usando multiagentes.
    Lê um arquivo CSV do diretório data/ e processa a pergunta do usuário.
    Esta ferramenta deve ser usada quando o usuário solicitar gráficos, visualizações ou análise visual.
    
    Args:
        csv_filename: Nome do arquivo CSV no diretório data/ (ex: 'dataset.csv')
        question: Pergunta ou objetivo da análise exploratória
    
    Returns:
        Resultado da análise EDA em formato JSON string
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
        
        # Verificar se é um arquivo CSV válido
        try:
            df = read_csv_robust(file_path)
            rows, cols = df.shape
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Erro ao ler o arquivo CSV: {str(e)}"
            })
        
        # Marcar que a EDA está rodando
        st.session_state.eda_running = True
        
        # Acionar o create_eda_crew - isso retorna a crew configurada
        eda_crew = create_eda_crew(str(file_path), question)
        
        # Executar a crew e obter o resultado
        result = eda_crew.kickoff()
        
        st.session_state.eda_running = False
        
        # Armazenar o markdown gerado no session_state
        st.session_state.eda_markdown = str(result)
        
        return json.dumps({
            "success": True,
            "result": str(result),
            "markdown_generated": True,
            "file_info": {
                "filename": csv_filename,
                "rows": rows,
                "columns": cols
            }
        })
        
    except Exception as e:
        st.session_state.eda_running = False
        return json.dumps({
            "success": False,
            "error": f"Erro ao executar análise EDA: {str(e)}"
        })

def limpar_pngs(diretorio):
    if not os.path.isdir(diretorio):
        print(f"Diretório não encontrado: {diretorio}")
        return

    arquivos_png = glob.glob(os.path.join(diretorio, "*.png"))
    for arquivo in arquivos_png:
        try:
            os.remove(arquivo)
            print(f"Removido: {arquivo}")
        except Exception as e:
            print(f"Erro ao remover {arquivo}: {e}")


# Criar o agente conversacional
agent = Agent(
    name="Einstein Data Scientist",
    instructions="""
### ROLE
Especialista em estatística, ciência de dados e inteligência artificial.
Atua como analista interativo, capaz de interpretar dados, gerar visualizações
e conduzir investigações exploratórias com profundidade.

### GOAL
Conversar com o usuário sobre seus dados e fornecer insights precisos. Você tem duas ferramentas à disposição:
1. analyze_csv_data: Para análises rápidas e perguntas diretas sobre os dados
2. execute_python_code: Para análises customizadas executando código Python
3. run_eda_analysis: Para análises exploratórias completas com visualizações (multiagentes)

### BACKSTORY
Você é Einstein, o cientista de dados! Sua missão é transformar dados em conhecimento acionável,
guiando o usuário com clareza, precisão e insights valiosos.

### REASONING - QUANDO USAR CADA FERRAMENTA:

**USE analyze_csv_data PARA:**
- Perguntas sobre estatísticas descritivas: média, mediana, desvio padrão, min, max
- Consultas sobre dimensões do dataset: quantas linhas, quantas colunas
- Verificação de valores: valores únicos, valores faltantes, tipos de dados
- Listagem de colunas disponíveis
- Visualização de primeiras linhas do dataset
- Contagens e frequências de valores
- Correlações numéricas (valores, não gráficos)
- Qualquer pergunta que possa ser respondida com números/texto

**USE execute_python_code PARA:**
- Análises customizadas que não são cobertas por analyze_csv_data
- Cálculos complexos envolvendo múltiplas colunas
- Filtragens e agregações específicas
- Transformações de dados
- Análises estatísticas avançadas
- Qualquer pergunta que exija código Python customizado
- Exemplo: "Quantos registros têm idade > 30 E salário < 5000?"
- Exemplo: "Qual a média de salário por categoria?"
- Exemplo: "Mostre os 10 maiores valores da coluna X"

**USE run_eda_analysis PARA:**
- Solicitações explícitas de gráficos: "mostre um gráfico", "crie uma visualização"
- Pedidos de plots específicos: histogram, boxplot, scatter plot, heatmap, bar chart
- Análise exploratória completa (EDA)
- Quando o usuário pedir "análise visual" ou "visualização"
- Quando múltiplos gráficos são necessários para responder adequadamente
- Quando a resposta requer interpretação visual dos padrões

### PROCESSO DE REASONING:
1. Analise a pergunta do usuário cuidadosamente
2. Identifique se a resposta requer visualização ou apenas números/texto
3. Se for pergunta simples → use analyze_csv_data (resposta em segundos)
4. Se a pergunta exigir calculo, pense no pedido feito, gere codigo python -> use execute_python_code
5. Se for pergunta visual → use run_eda_analysis (5-10 minutos)
6. Explique sua escolha ao usuário quando relevante

### COMO USAR execute_python_code:
- ATENCAO: O DataFrame já está carregado como 'df'
- Busque sempre validar o carregamento de csv
- Atente-se para os encodes e separadores possiveis em csv
- Use pandas, numpy, matplotlib e outras bibliotecas padrão
- SEMPRE use print() para mostrar resultados
- SEMPRE ao gerar gráficos salve os gráficos em /charts/conversa
- Exemplo de código:
  ```python
  print(df[df['idade'] > 30]['salario'].mean())
  ```
- Exemplo complexo:
  ```python
  resultado = df.groupby('categoria')['valor'].agg(['mean', 'sum', 'count'])
  print(resultado)
  ```

### ACT
- Seja conversacional, amigável e didático
- SEMPRE use analyze_csv_data primeiro para perguntas simples
- APENAS acione run_eda_analysis quando visualizações forem realmente necessárias
- Antes de usar run_eda_analysis, avise: "Eureka! Vou acionar os Multi Agentes para criar visualizações. Isso levará de 5 a 10 minutos..."
- Após receber resultados, explique os insights de forma clara
- Seja proativo em sugerir análises adicionais relevantes

### EXEMPLOS DE REASONING:

**Pergunta:** "Qual a média da coluna idade?"
**Reasoning:** Pergunta simples sobre estatística descritiva
**Ação:** Use analyze_csv_data com analysis_type='mean' e column_name='idade'

**Pergunta:** "Mostre a distribuição da coluna idade"
**Reasoning:** Requer visualização (histograma)
**Ação:** Use run_eda_analysis

**Pergunta:** "Quantas linhas tem o dataset?"
**Reasoning:** Pergunta sobre dimensões, resposta numérica simples
**Ação:** Use analyze_csv_data com analysis_type='shape'

**Pergunta:** "Crie um boxplot da coluna salário"
**Reasoning:** Solicitação explícita de gráfico
**Ação:** Use execute_python_code(python_code="plt.figure(figsize=(8, 6))
plt.boxplot(df[salário], labels=['Salario'])
plt.title('Boxplot de Salario')
plt.ylabel('Valores')
plt.grid(True)
plt.savefig('charts/conversa/boxplot_salario.png', dpi=300, bbox_inches='tight')

# Exibindo o gráfico
plt.show()
")

**Pergunta:** "Quais são as colunas disponíveis?"
**Reasoning:** Listagem simples, sem necessidade de visualização
**Ação:** Use analyze_csv_data com analysis_type='columns'

**Pergunta:** "Quais a média da coluna vendas?"
**Reasoning:** Listagem simples, sem necessidade de visualização
**Ação:** Use analyze_csv_data com analysis_type='columns'

**Pergunta:** "Faça uma análise exploratória completa"
**Reasoning:** EDA completa requer múltiplas visualizações
**Ação:** Use run_eda_analysis

**Pergunta:** "Quantos registros têm idade > 30?"
**Reasoning:** Requer filtragem customizada
**Ação:** execute_python_code(python_code="print(len(df[df['idade'] > 30]))")

**Pergunta:** "Qual a média de salário por categoria?"
**Reasoning:** Requer agregação por grupo
**Ação:** execute_python_code(python_code="print(df.groupby('categoria')['salario'].mean())")

**Pergunta:** "Quais são as 5 categorias com maior média de valor?"
**Reasoning:** Requer agregação, ordenação e seleção
**Ação:** execute_python_code(python_code="print(df.groupby('categoria')['valor'].mean().nlargest(5))")


### GUARDRAILS
- NUNCA acione run_eda_analysis para perguntas que podem ser respondidas com analyze_csv_data
- Seja claro sobre o tempo de processamento (5-10 minutos para EDA)
- Sempre confirme qual arquivo CSV está sendo analisado
- Use linguagem clara e evite jargões excessivos
- Explique os resultados de forma didática
- Quando usar execute_python_code, mostre o código executado ao usuário
""",
    model="gpt-4.1-mini",
    tools=[analyze_csv_data, execute_python_code, run_eda_analysis]
)


def get_or_create_eventloop():
    """
    Obtém o event loop atual ou cria um novo se não existir.
    Necessário para executar código assíncrono em threads do Streamlit.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def chat_with_agent_sync(user_message: str, session_id: str = "default_session") -> str:
    """
    Versão síncrona da função de chat para uso com Streamlit.
    Cria um event loop se necessário para lidar com operações assíncronas.
    
    Args:
        user_message: Mensagem do usuário
        session_id: ID da sessão para manter contexto
    
    Returns:
        Resposta do agente
    """
    try:
        # Obter ou criar event loop
        loop = get_or_create_eventloop()
        
        # Criar sessão com memória persistente
        session = SQLiteSession(session_id, "storage\conversacional\conversations.db")
        
        # Executar o agente de forma síncrona usando Runner.run_sync
        result = Runner.run_sync(
            agent,
            user_message,
            session=session

        )
        
        return result.final_output
        
    except Exception as e:
        # Se falhar, tentar sem sessão (sem memória)
        try:
            session = SQLiteSession(session_id)
            result = Runner.run_sync(
                agent,
                user_message,
                session=session
            )
            return result.final_output
        except Exception as e:
            try:
                session = SQLiteSession(session_id)
                result = Runner.run_sync(
                    agent,
                    user_message,
                    session=session
                )
                return result.final_output + "\n\n⚠️ (Executando sem memória de conversa)"
            except Exception as e2:
                return f"❌ Erro ao processar: {str(e)}\n\nTentativa alternativa: {str(e2)}"

col1, col2 = st.columns([1,10])  # Ajuste a proporção conforme necessário

primary_path = "eda_app/image/eistein_.jpg"
fallback_path = "image/eistein_.jpg"

# Verifica qual existe
if os.path.exists(primary_path):
    image = primary_path
elif os.path.exists(fallback_path):
    image = fallback_path
else:
    image = None  # ou um placeholder padrão


with col1:
    if image:
        st.image(image, use_container_width=True)
    else:
        st.warning("Imagem não encontrada em nenhum dos diretórios.")

with col2:
    st.title("Einstein Data Scientist - Chat")

# Configurar a página
st.set_page_config(
    page_title="Einstein Data Scientist - Chat",
    page_icon="🧠",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título
# st.title("🧠 Einstein Data Scientist")
st.markdown("*Converse com o agente sobre seus dados*")

# Sidebar
with st.sidebar:
    st.header("📁 Upload de Dados")
    
    uploaded_file = st.file_uploader(
        "Faça upload do seu arquivo CSV",
        type=["csv"],
        help="Selecione um arquivo CSV para análise"
    )
    
    if uploaded_file is not None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.selected_csv = uploaded_file.name
        st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
        
        try:
            df = read_csv_robust(file_path)
            st.subheader("📊 Preview dos Dados")
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"**Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    
    st.divider()
    
    st.header("ℹ️ Informações")
    st.markdown("""
    ### Como usar:
    1. 📤 Faça upload do arquivo CSV
    2. 💬 Converse com Einstein sobre os dados
    3. 📊 Solicite gráficos e visualizações
    4. ⏱️ Aguarde 5-10 min para análises visuais
    
    ### Exemplos de perguntas:
    - "Qual a média da coluna X?"
    - "Mostre um histograma da coluna Y"
    - "Crie um gráfico de correlação"
    - "Faça uma análise exploratória completa"
    """)
    
    st.divider()
    
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Inicializar histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user_session"

if "crew_logs" not in st.session_state:
    st.session_state.crew_logs = []

# Área de chat
st.subheader("💬 Chat com Einstein")

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua mensagem..."):
    if st.session_state.selected_csv is None:
        st.warning("⚠️ Por favor, faça upload de um arquivo CSV primeiro!")
    else:
        enhanced_prompt = f"{prompt}\n\n[Contexto: O arquivo CSV atual é '{st.session_state.selected_csv}']"
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Obter resposta do agente
        with st.chat_message("assistant"):
            # Verificar se vai acionar EDA (palavras-chave)
            keywords = ["gráfico", "grafico", "visualiza", "plot", "boxplot", "histogram", 
                       "heatmap", "scatter", "correlação", "correlacao", "mostre", "crie",
                       "eda", "exploratória", "exploratoria", "completa", "análise visual"]
            
            will_run_eda = any(keyword in prompt.lower() for keyword in keywords)
            
            # Inicializar área de logs
            if "crew_logs" not in st.session_state:
                st.session_state.crew_logs = []       
            if will_run_eda:
                # Aviso inicial ao usuário ANTES de qualquer processamento
                st.info("🚀 **Acionando Fluxo Multiagente para Análise Visual**")
                st.warning("⏱️ **Tempo estimado:** 5 a 10 minutos para análise completa")
                st.markdown("---")
                
                # Criar área de logs em tempo real
                st.markdown("### 📋 Logs de Execução da Crew")
                logs_container = st.container()
                logs_placeholder = st.empty()
                
                # Limpar logs anteriores
                st.session_state.crew_logs = []
                
                # Função para adicionar log
                def add_log(message):
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {message}"

                    # Inicializar área de logs
                    if "crew_logs" not in st.session_state:
                        st.session_state.crew_logs = []   

                    st.session_state.crew_logs.append(log_entry)
                    
                    # Atualizar display de logs
                    with logs_placeholder.container():
                        st.text_area(
                            "Logs em tempo real:",
                            value="\n".join(st.session_state.crew_logs[-20:]),  # Últimas 20 linhas
                            height=200,
                            key=f"logs_display_{len(st.session_state.crew_logs)}"
                        )
                
                # Log inicial
                add_log("🚀 Iniciando fluxo multiagente...")
                add_log(f"📁 Arquivo: {st.session_state.selected_csv}")
                add_log(f"❓ Pergunta: {prompt}")
                
                st.markdown("---")
                
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.markdown("### 🤖 Multi Agentes em Ação")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()
                    
                    import time
                    
                    # Variável para controlar o progresso
                    progress_data = {"current": 0, "running": True}
                    
                    def update_progress():
                        """Atualiza a barra de progresso de forma realista"""
                        start_time = time.time()
                        stages = [
                            (0, 15, "🔍 Carregando e validando dados..."),
                            (15, 30, "📊 Analisando estrutura e tipos de dados..."),
                            (30, 45, "📈 Calculando estatísticas descritivas..."),
                            (45, 60, "🎨 Gerando visualizações (histogramas, boxplots)..."),
                            (60, 75, "🔗 Analisando correlações e relações..."),
                            (75, 85, "📉 Criando gráficos de distribuição..."),
                            (85, 95, "📝 Compilando relatório final..."),
                            (95, 98, "✨ Finalizando análise...")
                        ]
                        
                        stage_idx = 0
                        last_logged_stage = -1
                        
                        while progress_data["running"] and stage_idx < len(stages):
                            start_pct, end_pct, message = stages[stage_idx]
                            
                            # Adicionar log quando entrar em novo estágio
                            if stage_idx != last_logged_stage:
                                add_log(message)
                                last_logged_stage = stage_idx
                            
                            # Incrementar progresso gradualmente dentro do estágio
                            for pct in range(start_pct, end_pct + 1):
                                if not progress_data["running"]:
                                    break
                                
                                progress_data["current"] = pct
                                progress_bar.progress(pct / 100)
                                status_text.markdown(f"**{message}**")
                                
                                # Calcular tempo decorrido
                                elapsed = int(time.time() - start_time)
                                mins, secs = divmod(elapsed, 60)
                                time_text.text(f"⏱️ Tempo decorrido: {mins}min {secs}s")
                                
                                # Velocidade variável: mais lento no início, mais rápido no fim
                                if pct < 30:
                                    time.sleep(4)  # Mais lento no início
                                elif pct < 60:
                                    time.sleep(3)  # Médio
                                elif pct < 85:
                                    time.sleep(2.5)  # Mais rápido
                                else:
                                    time.sleep(2)  # Rápido no final
                            
                            stage_idx += 1
                        
                        # Manter em 98% até a conclusão real
                        while progress_data["running"]:
                            time.sleep(1)
                    
                    # Iniciar thread de progresso
                    progress_thread = threading.Thread(target=update_progress, daemon=True)
                    progress_thread.start()
                    
                    try:
                        add_log("🔄 Executando agente conversacional...")
                        add_log("🤖 Agente está processando a solicitação...")
                        
                        # Executar o agente (isso vai chamar create_eda_crew e kickoff)
                        response = chat_with_agent_sync(
                            enhanced_prompt, 
                            st.session_state.session_id
                        )
                        
                        # Sinalizar conclusão
                        progress_data["running"] = False
                        progress_thread.join(timeout=2)
                        
                        add_log("✅ Crew executada com sucesso!")
                        add_log("📊 Processando resultados...")
                        
                        # Completar progresso
                        progress_bar.progress(100)
                        status_text.markdown("**✅ Análise multiagente concluída com sucesso!**")
                        time_text.text("")
                        
                        add_log("✨ Análise completa! Exibindo resultados...")
                        time.sleep(2)
                        
                    except Exception as e:
                        progress_data["running"] = False
                        add_log(f"❌ ERRO: {str(e)}")
                        response = f"❌ Erro ao processar: {str(e)}"
                    finally:
                        # Limpar barra de progresso após 2 segundos
                        time.sleep(1)
                        progress_placeholder.empty()
                        add_log("📋 Logs finalizados.")
                
                # Exibir resposta do agente
                st.markdown(response)
                
                # Aviso sobre onde encontrar o relatório
                if st.session_state.eda_markdown:
                    st.success("✅ Relatório completo disponível abaixo do chat!")
                    st.info("👇 Role para baixo para ver o relatório detalhado e os gráficos gerados.")
            else:
                # Resposta rápida sem barra de progresso
                with st.spinner("Einstein está pensando..."):
                    try:
                        response = chat_with_agent_sync(
                            enhanced_prompt, 
                            st.session_state.session_id
                        )
                    except Exception as e:
                        response = f"❌ Erro ao processar: {str(e)}"
                
                # Exibir resposta
                st.markdown(response)
            
            # Adicionar resposta ao histórico (para ambos os casos)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })

# Área persistente para exibir graficso simples gerados pelo assistente de conversa
st.divider()
if "mostrar_imagem" not in st.session_state:
    st.session_state.mostrar_imagem = False

charts_dir_conv = Path("charts/conversa")
if charts_dir_conv.exists():
    image_extensions = ['.png', '.jpg', '.jpeg', '.svg']
    chart_files = []
    
    for ext in image_extensions:
        chart_files.extend(list(charts_dir_conv.glob(f"*{ext}")))
    
    if chart_files:
        chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        st.success(f"✅ {len(chart_files)} gráfico(s) disponível(is) para download")
        
        # Exibir gráficos em grid 2x2
        cols_per_row = 2
        for i in range(0, len(chart_files), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(chart_files):
                    chart_file = chart_files[idx]
                    
                    with col:
                        # Exibir imagem
                        try:

                            st.image(
                                str(chart_file),
                                use_container_width=True,
                                caption=chart_file.name
                            )
                            st.session_state.mostrar_imagem = True
                            # Botão de download individual
                            with open(chart_file, "rb") as f:
                                st.download_button(
                                    label=f"⬇️ Baixar {chart_file.name}",
                                    data=f.read(),
                                    file_name=chart_file.name,
                                    mime=f"image/{chart_file.suffix[1:]}",
                                    key=f"download_chart_{chart_file.name}",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Erro ao carregar {chart_file.name}: {e}")
        if st.button("Limpar Imagem"):
            st.session_state.mostrar_imagem = False
            limpar_pngs(charts_dir_conv)

    else:
        st.info("ℹ️ Nenhum gráfico disponível.")
else:
    Path("charts/conversa").mkdir(parents=True, exist_ok=True)
    #st.warning("⚠️ Diretório 'charts/conversa' não encontrado.")


# Área persistente para exibir relatório EDA (fora do chat)
st.divider()

if st.session_state.eda_markdown:
    st.markdown("## 📊 Relatório de Análise Exploratória de Dados")
    st.info("💡 Este relatório permanece visível até que você clique em 'Limpar Relatório'")
    
    # Botão de limpar no topo
    col_clear1, col_clear2 = st.columns([3, 1])
    with col_clear2:
        if st.button("🗑️ Limpar Relatório", key="clear_report_top", type="primary"):
            st.session_state.eda_markdown = None
            st.session_state.crew_logs = []
            st.rerun()
    
    st.markdown("---")
    
    # Container para o relatório
    with st.container():
        # Processar markdown para incluir imagens locais
        markdown_content = st.session_state.eda_markdown
        
        # Substituir caminhos relativos de imagens por caminhos absolutos
        import re
        charts_dir = Path("charts")
        
        if charts_dir.exists():
            # Encontrar todas as referências de imagens no markdown
            def replace_image_path(match):
                alt_text = match.group(1)
                img_path = match.group(2)
                
                # Se for caminho relativo, converter para absoluto
                if not img_path.startswith(('http://', 'https://', 'data:')):
                    # Remover prefixo charts/ se existir
                    img_filename = img_path.split('/')[-1]
                    full_path = charts_dir / img_filename
                    
                    if full_path.exists():
                        return f"![{alt_text}]({full_path.as_posix()})"
                
                return match.group(0)
            
            markdown_content = re.sub(
                r'!\[(.*?)\]\((.*?)\)',
                replace_image_path,
                markdown_content
            )
        
        # Exibir o markdown processado
        st.markdown(markdown_content, unsafe_allow_html=True)
        
        # Seção de download e ações
        st.markdown("---")
        st.markdown("### 📥 Downloads e Ações")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Botão para download do relatório em markdown
            st.download_button(
                label="📄 Baixar Relatório (Markdown)",
                data=st.session_state.eda_markdown,
                file_name="relatorio_eda.md",
                mime="text/markdown",
                key="download_markdown_report",
                use_container_width=True
            )
        
        with col2:
            # Botão para exibir logs
            if st.session_state.crew_logs:
                with st.expander("📋 Ver Logs de Execução"):
                    st.text_area(
                        "Logs completos:",
                        value="\n".join(st.session_state.crew_logs),
                        height=300,
                        key="logs_final_display"
                    )
        
        with col3:
            # Botão para limpar relatório
            if st.button("🗑️ Limpar Tudo", key="clear_report_bottom", use_container_width=True):
                st.session_state.eda_markdown = None
                st.session_state.crew_logs = []
                st.rerun()
        
        # Seção de gráficos para download
        st.markdown("---")
        st.markdown("### 📊 Gráficos Gerados")
        
        charts_dir = Path("charts")
        if charts_dir.exists():
            image_extensions = ['.png', '.jpg', '.jpeg', '.svg']
            chart_files = []
            
            for ext in image_extensions:
                chart_files.extend(list(charts_dir.glob(f"*{ext}")))
            
            if chart_files:
                chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                st.success(f"✅ {len(chart_files)} gráfico(s) disponível(is) para download")
                
                # Exibir gráficos em grid 2x2
                cols_per_row = 2
                for i in range(0, len(chart_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(chart_files):
                            chart_file = chart_files[idx]
                            
                            with col:
                                # Exibir imagem
                                try:
                                    st.image(
                                        str(chart_file),
                                        use_container_width=True,
                                        caption=chart_file.name
                                    )
                                    
                                    # Botão de download individual
                                    with open(chart_file, "rb") as f:
                                        st.download_button(
                                            label=f"⬇️ Baixar {chart_file.name}",
                                            data=f.read(),
                                            file_name=chart_file.name,
                                            mime=f"image/{chart_file.suffix[1:]}",
                                            key=f"download_chart_{chart_file.name}",
                                            use_container_width=True
                                        )
                                except Exception as e:
                                    st.error(f"Erro ao carregar {chart_file.name}: {e}")
            else:
                st.info("ℹ️ Nenhum gráfico disponível.")
        else:
            st.warning("⚠️ Diretório 'charts' não encontrado.")

st.divider()
st.caption("🧠 Einstein Data Scientist - Powered by OpenAI Agents SDK")