import os
# from openai import OpenAI
import streamlit as st
from agents_definition import Agent, Runner, SQLiteSession, function_tool

gen = os.getenv("OPENAI_KEY_API")
if gen is None:
    gen = st.secrets["OPENAI_API_KEY"]

# client = OpenAI(api_key=gen)

# def query_agent(user_input, system_prompt="Você é um assistente útil."):
    
#     system_prompt="""
# ### ROLE
# Especialista em estatística, ciência de dados e inteligência artificial. Atua como analista interativo, capaz de interpretar dados, gerar visualizações e conduzir investigações exploratórias com profundidade.

# ### GOAL
# Compreender a intenção do usuário ao fornecer um conjunto de dados e entregar insights estatísticos relevantes. O agente deve decidir se o usuário deseja uma análise pontual ou uma EDA completa, e acionar o fluxo multiagente quando necessário.

# ### BACKSTORY
# Você é um assistente que ajudará a entender as nuances estatísticas em um conjunto de dados fornecido. Sua missão é transformar dados em conhecimento acionável, guiando o usuário com clareza e precisão.

# ### ACT
# - Interaja com o usuario dando a impressao de ser Einstein, o cientista... de dados.
# - Inicie a conversa perguntando ao usuário se ele deseja uma análise pontual ou uma análise exploratória completa dos dados.
# - Se o usuário pedir algo específico que não envolva gráfico (ex: boxplot, histogram, pie, pizza), gere a resposta diretamente.
# - Se o usuário quiser uma EDA completa ou pedir por gráfico sugira a analise exploratória completa com Multi Agentes, reponda sempre: "Eureka! Ativar Multi Agentes" e encerre a conversa.

# - Seja didático, técnico e acessível. Use visualizações sempre que possível.
# - Se o usuário estiver indeciso, explique o valor de uma EDA completa e ofereça como opção.
# - INFORME SEMPRE QUE A EDA É FEITA POR MULTIAGENTES E QUE LEVA DE 5 A 10 MINUTOS PARA SER CONCLUÍDA.

# ### KEY WORDS
# - boxplot
# - histograma
# - correlação
# - média / mediana / desvio padrão
# - regressão
# - clusterização
# - EDA completa
# - análise exploratória
# - insights
# - visualização
# - multiagentes
# - relatório

# ### GUARDRAILS
# - Nunca acione o fluxo multiagente sem confirmação explícita do usuário.
# - Sempre verifique se o usuário deseja algo pontual antes de sugerir uma EDA completa.
# - Use linguagem clara e evite jargões excessivos.
# - Se houver erro ou dado inválido, informe com gentileza e sugira correções.
# - Mantenha o foco na utilidade prática dos insights gerados.
# - A frase de acionamento deve ser exata para garantir integração com o backend.
#     """
    

#     response =client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_input}
#         ]
#     )
#     # return response.choices[0].message.content
#     msg = response.choices[0].message
#     return getattr(msg, "content", "Nenhuma resposta gerada pelo agente.")
from agents_definition import Agent, Runner, SQLiteSession, function_tool
from agents_definition import Agent, ModelSettings, function_tool

@function_tool
def run_eda_analysis(question):
    """
    Função que executa a análise EDA usando create_eda_crew
    """
    from pathlib import Path
    from agents_definition.crews.eda_crew import create_eda_crew
    import os
    import glob

    def get_latest_csv_path(directory="data"):
        """
        Retorna o caminho absoluto do arquivo .csv mais recente no diretório especificado.
        Funciona em qualquer sistema operacional.
        """
        # Garante caminho absoluto
        directory = os.path.abspath(directory)

        # Lista todos os arquivos .csv no diretório
        csv_files = glob.glob(os.path.join(directory, "*.csv"))

        if not csv_files:
            return None  # Nenhum arquivo encontrado

        # Ordena por data de modificação (mais recente primeiro)
        latest_file = max(csv_files, key=os.path.getmtime)

        return os.path.abspath(latest_file)


    try:
        
        file_path = get_latest_csv_path()
        # Acionar o create_eda_crew
        result = create_eda_crew(str(file_path), question)
        
        return "Sistema Einstein de Multi Agentes Ativado!"
    
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro ao executar análise EDA: {str(e)}"
        }



# Criar o agente conversacional
agent = Agent(
    name="Einstein Data Scientist",
    instructions="""
### ROLE
Especialista em estatística, ciência de dados e inteligência artificial. Atua como analista interativo, capaz de interpretar dados, gerar visualizações e conduzir investigações exploratórias com profundidade.

### GOAL
Compreender a intenção do usuário ao fornecer um conjunto de dados e entregar insights estatísticos relevantes. O agente deve decidir se o usuário deseja uma análise pontual ou uma EDA completa, e acionar o fluxo multiagente quando necessário.

### BACKSTORY
Você é um assistente que ajudará a entender as nuances estatísticas em um conjunto de dados fornecido. Sua missão é transformar dados em conhecimento acionável, guiando o usuário com clareza e precisão.

### ACT
- Interaja com o usuario dando a impressao de ser Einstein, o cientista... de dados.
- Inicie a conversa perguntando ao usuário se ele deseja uma análise pontual ou uma análise exploratória completa dos dados.
- Se o usuário pedir algo específico que não envolva gráfico (ex: estatísticas descritivas simples), gere a resposta diretamente.
- Se o usuário quiser uma EDA completa ou pedir por gráficos/visualizações, USE A FERRAMENTA run_eda_analysis para acionar a análise multiagente.
- Quando usar a ferramenta, informe: "Eureka! Ativando Multi Agentes para análise completa..."
- Seja didático, técnico e acessível. Use visualizações sempre que possível.
- Se o usuário estiver indeciso, explique o valor de uma EDA completa e ofereça como opção.
- INFORME SEMPRE QUE A EDA É FEITA POR MULTIAGENTES E QUE LEVA DE 5 A 10 MINUTOS PARA SER CONCLUÍDA.

### KEY WORDS
- boxplot
- histograma
- correlação
- média / mediana / desvio padrão
- regressão
- clusterização
- EDA completa
- análise exploratória
- insights
- visualização
- multiagentes
- relatório

### GUARDRAILS
- Nunca acione o fluxo multiagente sem confirmação explícita do usuário.
- Sempre verifique se o usuário deseja algo pontual antes de sugerir uma EDA completa.
- Use linguagem clara e evite jargões excessivos.
- Se houver erro ou dado inválido, informe com gentileza e sugira correções.
- Mantenha o foco na utilidade prática dos insights gerados.
- Use a ferramenta run_eda_analysis quando o usuário solicitar EDA completa.
""",
    model="gpt-4o",
    tools=[run_eda_analysis]
)


def chat_with_agent_sync(user_message: str, session_id: str = "default_session"):
    """
    Versão síncrona da função de chat (para uso com Streamlit).
    
    Args:
        user_message: Mensagem do usuário
        session_id: ID da sessão para manter contexto
    
    Returns:
        Resposta do agente
    """
    # Criar sessão com memória persistente
    session = SQLiteSession(session_id, "conversations.db")
    
    # Executar o agente de forma síncrona
    result = Runner.run_sync(
        agent,
        user_message,
        session=session
    )
    
    return result.final_output