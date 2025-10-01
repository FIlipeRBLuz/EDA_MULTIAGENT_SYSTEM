# Sistema de Análise Exploratória de Dados com Agentes CrewAI

Um sistema multiagente avançado para análise exploratória de dados (EDA) que utiliza CrewAI, Python e Streamlit. O sistema é capaz de analisar qualquer arquivo CSV e gerar relatórios completos com visualizações e insights automatizados.

## Funcionalidades

###  **Equipe de Agentes Especializados**
- **Descritor de Dados**: Analisa tipos, estatísticas e valores ausentes
- **Detector de Padrões**: Identifica distribuições e tendências
- **Detector de Anomalias**: Encontra outliers e valores atípicos
- **Analisador de Relações**: Mapeia correlações entre variáveis
- **Especialista em Visualizações**: Cria gráficos personalizados
- **Agente de Conclusão**: Sintetiza descobertas e gera insights

###  **Análises Automáticas**
- ✅ Detecção automática de separadores CSV
- ✅ Análise de tipos de dados (numéricos, categóricos)
- ✅ Distribuições com histogramas e boxplots
- ✅ Medidas de tendência central e variabilidade
- ✅ Detecção de padrões e valores frequentes
- ✅ Identificação de outliers com visualizações
- ✅ Análise de correlação com heatmap
- ✅ Gráficos personalizados sob demanda

###  **Relatórios Profissionais**
- ✅ Relatório markdown completo com gráficos incorporados
- ✅ Download em PDF com texto e imagens
- ✅ Interface web interativa
- ✅ Visualizações integradas

##  Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone ou baixe o projeto**
   ```bash
   # Se você tem o arquivo zip, extraia-o
   # Ou clone o repositório se disponível
   ```

2. **Navegue até o diretório do projeto**
   ```bash
   cd eda_app
   ```

3. **Crie um ambiente virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # No Windows:
   venv\Scripts\activate
   
   # No macOS/Linux:
   source venv/bin/activate
   ```

4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure sua chave da OpenAI**
   
   Você precisa de uma chave da API da OpenAI. Crie um arquivo `.env` na raiz do projeto:

   ''' crie sua variavel de ambiente OPENAI_KEY_API e insira sua key da plataforma como valor '''
   ''' Crie a CHROMA_OPENAI_API_KEY como variavel de ambiente, o valor é o mesmo da OPENAI_KEY_API '''

##  Como Executar

1. **Inicie a aplicação**
   ```bash
   streamlit run app.py
   ```

2. **Acesse no navegador**
   - A aplicação abrirá automaticamente em `http://localhost:8501`
   - Se não abrir automaticamente, acesse o link manualmente

##  Como Usar

### 1. **Upload do Arquivo**
- Clique em "Escolha um arquivo CSV"
- Selecione seu arquivo CSV (qualquer separador é suportado)
- Visualize o preview dos dados

### 2. **Faça sua Pergunta**
- Digite uma pergunta específica sobre seus dados
- Exemplos:
  - "Quais são as principais correlações entre as variáveis?"
  - "Existem outliers significativos nos dados?"
  - "Qual a distribuição das variáveis numéricas?"

### 3. **Solicite Gráficos Personalizados (Opcional)**
- Use o campo "Gráfico Personalizado"
- Exemplos:
  - "Crie um gráfico de dispersão entre idade e salário"
  - "Faça um gráfico de barras das categorias mais frequentes"
  - "Gere um histograma da distribuição de vendas"

### 4. **Analise os Resultados**
- Clique em "Iniciar Análise Completa"
- Aguarde o processamento (pode levar alguns minutos)
- Visualize o relatório completo
- Baixe o PDF do relatório

## Estrutura do Projeto

```
eda_app/
├── app.py                          # Aplicação principal Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
├── .env                           # Variáveis de ambiente (criar)
├── agents/                        # Módulo dos agentes
│   ├── __init__.py
│   ├── crews/
│   │   └── eda_crew.py           # Definição da equipe de agentes
│   └── tools/
│       └── eda_tools.py          # Ferramentas dos agentes
├── data/                          # Dados de exemplo
│   └── exemplo_vendas.csv
├── charts/                        # Gráficos gerados
├── reports/                       # Relatórios gerados
└── eda_app/                      # Diretórios de trabalho
    ├── charts/
    ├── data/
    └── reports/
```

## Tipos de Gráficos Suportados

O sistema pode criar automaticamente:

- **Gráficos de Dispersão**: Para analisar correlações entre variáveis numéricas
- **Gráficos de Barras**: Para variáveis categóricas e contagens
- **Histogramas**: Para distribuições de variáveis numéricas
- **Boxplots**: Para identificar outliers e quartis
- **Gráficos de Pizza**: Para proporções de categorias
- **Gráficos de Linha**: Para dados temporais ou sequenciais
- **Heatmaps**: Para matrizes de correlação

## Solução de Problemas

### Erro de Chave da API
```
BadRequestError: OpenAIException - Error code: 400
```
**Solução**: Verifique se sua chave da OpenAI está configurada corretamente.

### Erro de Dependências
```
ModuleNotFoundError: No module named 'crewai'
```
**Solução**: Execute `pip install -r requirements.txt` novamente.

### Erro de Permissão de Arquivo
**Solução**: Certifique-se de que o Python tem permissão para criar arquivos no diretório.

### Problemas com PDF
Se a geração de PDF falhar, o sistema automaticamente oferece download em formato Markdown.

## Contribuições

Este projeto foi desenvolvido como uma demonstração de sistema multiagente para análise de dados. Sinta-se à vontade para:

- Reportar bugs
- Sugerir melhorias
- Adicionar novos tipos de análise
- Melhorar as visualizações

## Licença

Este projeto é fornecido como está, para fins educacionais e de demonstração.

## Suporte

Se encontrar problemas:

1. Verifique se todas as dependências estão instaladas
2. Confirme se a chave da OpenAI está configurada
3. Certifique-se de que está usando Python 3.8+
4. Verifique se há mensagens de erro no terminal

---

**Desenvolvido com CrewAI, Streamlit e Python**
