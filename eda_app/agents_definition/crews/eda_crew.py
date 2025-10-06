from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
import os
from ..tools.eda_tools import describe_data, plot_distributions, plot_correlations, detect_outliers, create_custom_chart, execute_python_code
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

gen = os.getenv("OPENAI_KEY_API")
if gen is None:
    gen = st.secrets["OPENAI_API_KEY"]

# Configurar o modelo LLM
llm = LLM(model="gpt-4.1-mini",api_key=gen)

# Agentes especializados

data_describer = Agent(
    role='Descritor de Dados',
    goal='Descrever a estrutura fundamental dos dados, incluindo tipos, estatísticas e valores ausentes.',
    backstory='Você é um especialista em análise de dados focado em fornecer uma visão geral clara e concisa de qualquer conjunto de dados.',
    tools=[describe_data],
    llm=llm,
    allow_delegation=False,
    verbose=True,
    reasoning=False
)

pattern_detector = Agent(
    role='Detector de Padrões e Tendências',
    goal='Identificar padrões, tendências e distribuições nos dados.',
    backstory='Você é um analista de dados com um olhar aguçado para tendências e padrões. Você é especialista em visualização de dados para revelar insights.',
    tools=[plot_distributions, execute_python_code],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

anomaly_detector = Agent(
    role='Detector de Anomalias',
    goal='Encontrar outliers e anomalias nos dados.',
    backstory='Você é um detetive de dados, especializado em encontrar valores atípicos que podem distorcer uma análise.',
    tools=[detect_outliers, execute_python_code],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

relationship_analyzer = Agent(
    role='Analisador de Relações',
    goal='Analisar e visualizar as relações entre as variáveis.',
    backstory='Você é um especialista em modelagem de dados que entende como as variáveis interagem e influenciam umas às outras.',
    tools=[plot_correlations],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

custom_chart_agent = Agent(
    role='Especialista em Visualizações Personalizadas',
    goal='Criar gráficos personalizados baseados nas solicitações específicas do usuário.',
    backstory='Você é um especialista em visualização de dados que pode interpretar solicitações de gráficos e gerar código Python para criar visualizações customizadas.',
    tools=[create_custom_chart, execute_python_code],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

llm_reasoning = LLM(model="gpt-4o",api_key=gen)
conclusion_agent = Agent(
    role='Agente de Conclusão e Insights',
    goal='Sintetizar as descobertas de outros agentes e fornecer um resumo conclusivo e insights.',
    backstory='Você é um estrategista de dados que pode ver o quadro geral. Sua força reside em transformar análises complexas em insights acionáveis e responder perguntas específicas do usuário.',
    tools=[],
    llm=llm_reasoning,
    reasonming=True,
    allow_delegation=True,
    verbose=True
)

# conversational_agent = Agent(
#     role="Assistente Conversacional",
#     goal="Conversar com o usuário, entender o que ele precisa e acionar outros agentes quando necessário.",
#     backstory="Um agente simpático e inteligente que atua como facilitador, coordenando especialistas para ajudar o usuário.",
#     allow_delegation=True,       # <<< 🔥 Pode delegar tarefas
#     verbose=True,
#     llm=llm_reasoning
# )

# # Associar especialistas ao agente principal
# conversational_agent.crew = [data_describer, pattern_detector, anomaly_detector, relationship_analyzer, custom_chart_agent,llm_reasoning]

# Função para criar a equipe de EDA

def create_eda_crew(csv_path, user_question):
    import os
    # Configure custom storage location
    custom_storage_path = "./storage"
    os.makedirs(custom_storage_path, exist_ok=True)


    # Tarefas
# Task base de conversa
    # conversation_task = Task(
    #     description="Entender a necessidade do usuário e responder de forma natural.",
    #     expected_output='Texto de resposta a pergunta do cliente',
    #     agent=conversational_agent
    # )

    describe_task = Task(
        description=f'Use a ferramenta describe_data para analisar o arquivo CSV em {csv_path}. Forneça uma descrição completa dos dados.',
        agent=data_describer,
        expected_output='Uma descrição detalhada em markdown dos tipos de dados, estatísticas e valores ausentes.'
    )

    patterns_task = Task(
        description=f'Use a ferramenta plot_distributions para analisar e plotar as distribuições de todas as variáveis no arquivo CSV em {csv_path}. A ferramenta retornará um relatório markdown completo com gráficos incorporados e análises detalhadas.',
        agent=pattern_detector,
        expected_output='Um relatório markdown completo com gráficos de distribuição incorporados e análises estatísticas detalhadas para cada variável.'
    )

    outliers_task = Task(
        description=f'Use a ferramenta detect_outliers para detectar e plotar outliers para todas as variáveis numéricas no arquivo CSV em {csv_path}. A ferramenta retornará um relatório markdown completo com gráficos incorporados.',
        agent=anomaly_detector,
        expected_output='Um relatório markdown completo sobre outliers com gráficos incorporados, estatísticas detalhadas e interpretações.'
    )

    correlation_task = Task(
        description=f'Use a ferramenta plot_correlations para analisar e plotar as correlações entre as variáveis numéricas no arquivo CSV em {csv_path}. A ferramenta retornará um relatório markdown completo com heatmap incorporado.',
        agent=relationship_analyzer,
        expected_output='Um relatório markdown completo de correlações com heatmap incorporado e análise detalhada das relações entre variáveis.'
    )

    custom_chart_task = Task(
        description=f'Use a ferramenta create_custom_chart para criar um gráfico personalizado baseado na pergunta do usuário: "{user_question}". Analise a solicitação e crie uma visualização apropriada usando os dados do arquivo {csv_path}.',
        agent=custom_chart_agent,
        expected_output='Um relatório markdown com o gráfico personalizado criado, incluindo o código Python gerado e explicação da visualização.'
    )

    conclusion_task = Task(
        description=f'''Compile um relatório final completo e detalhado em markdown que integre TODOS os relatórios dos outros agentes. O relatório deve:
        
        1. **Cabeçalho**: Título profissional e data da análise
        2. **Resumo Executivo**: Principais descobertas em formato executivo
        3. **Resposta à Pergunta**: Responda especificamente: "{user_question}"
        4. **Incorporar TODOS os relatórios dos agentes**:
           - Copie integralmente o relatório de descrição dos dados
           - Copie integralmente o relatório de distribuições (COM os gráficos incorporados)
           - Copie integralmente o relatório de outliers (COM os gráficos incorporados)
           - Copie integralmente o relatório de correlações (COM os gráficos incorporados)
           - Copie integralmente o relatório de gráfico personalizado (COM o gráfico incorporado)
        5. **Insights Finais**: Síntese das descobertas mais importantes
        6. **Recomendações**: Próximos passos sugeridos
        
        **CRÍTICO**: 
        - MANTENHA todas as referências de imagens (![...](eda_app/charts/...)) dos outros agentes
        - NÃO remova ou modifique os caminhos das imagens
        - Organize o conteúdo de forma hierárquica e profissional
        - Use formatação markdown consistente
        
        O resultado deve ser um documento único e completo com TODOS os gráficos incorporados.''',
        agent=conclusion_agent,
        context=[describe_task, patterns_task, outliers_task, correlation_task, custom_chart_task],
        expected_output='Um relatório final completo em markdown que incorpora TODOS os relatórios dos agentes com gráficos incluídos, formatado profissionalmente.'
    )

    # manager = Agent(
    # role="Project Manager",
    # goal="Coordenar a equipe eficientemente acionando os agentes necessarios de acordo com a pergunta do cliente",
    # backstory="Gerente experiente em projetos complexos de ciência de dados e data analytics",
    # allow_delegation=True,
    # reasoning=True
    # )


    # Crew
    eda_crew = Crew(
        agents=[data_describer, pattern_detector, anomaly_detector, relationship_analyzer, custom_chart_agent, conclusion_agent],
        tasks=[describe_task, patterns_task, outliers_task, correlation_task, custom_chart_task, conclusion_task],
        process="sequential",
        verbose=True,
        memory=True,
        planning=False,
        planning_llm='gpt-4o',
        #manager_agent=manager,
        long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path=f"{custom_storage_path}/memory.db"
            )
        )
    )

    return eda_crew 
