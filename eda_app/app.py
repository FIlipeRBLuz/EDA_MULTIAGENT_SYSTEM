import streamlit as st
import pandas as pd
from agents.crews.eda_crew import create_eda_crew
import os
from datetime import datetime
import re
import base64
import pdfkit
from markdown import markdown
from PIL import Image


image = "image/eistein_.jpg"

st.set_page_config(
    page_title="Challenge Accepted - EDA MULTI AGENT",
    layout="wide"
)

# def create_download_link(markdown_content, filename):
#     """Cria um link de download para o conteúdo markdown como PDF"""
#     import re
#     import os

#     try:
#         # Criar diretórios se não existirem
#         os.makedirs("reports", exist_ok=True)
        
#         # Salvar o markdown em um arquivo temporário
#         md_path = f"reports/{filename}.md"
#         pdf_path = f"reports/{filename}.pdf"
        
#         # Processar o markdown para caminhos absolutos das imagens
        
#         current_dir = os.getcwd().replace("\\", "/")
#         processed_content = re.sub(
#             r'!\[(.*?)\]\(charts/(.*?)\)',
#             rf'![\1]({current_dir}/charts/\2)',
#             markdown_content
#         )
        
#         with open(md_path, "w", encoding="utf-8") as f:
#             f.write(processed_content)
        
#         # Tentar converter para PDF usando diferentes métodos
#         conversion_success = False
        
#         # Método 1: md-to-pdf
#         result = os.system(f"md-to-pdf {md_path} {pdf_path}")
#         if result == 0 and os.path.exists(pdf_path):
#             conversion_success = True
#         else:
#             # Método 2: weasyprint (fallback)
#             try:
#                 import weasyprint
#                 from markdown import markdown
                
#                 # Converter markdown para HTML
#                 html_content = f"""
#                 <!DOCTYPE html>
#                 <html>
#                 <head>
#                     <meta charset="utf-8">
#                     <style>
#                         body {{ font-family: Arial, sans-serif; margin: 40px; }}
#                         h1, h2, h3 {{ color: #1f77b4; }}
#                         img {{ max-width: 100%; height: auto; }}
#                     </style>
#                 </head>
#                 <body>
#                 {markdown(processed_content)}
#                 </body>
#                 </html>
#                 """
                
#                 # Gerar PDF com weasyprint
#                 weasyprint.HTML(string=html_content).write_pdf(pdf_path)
#                 conversion_success = True
#             except Exception as e:
#                 st.error(f"Erro na conversão alternativa: {str(e)}")
        
#         # Verificar se o PDF foi criado
#         if conversion_success and os.path.exists(pdf_path):
#             with open(pdf_path, "rb") as f:
#                 pdf_data = f.read()
            
#             b64_pdf = base64.b64encode(pdf_data).decode()
#             href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}.pdf" style="background-color: #1f77b4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📄 Baixar Relatório em PDF</a>'
#             return href
#         else:
#             # Oferecer download do markdown como alternativa
#             b64_md = base64.b64encode(markdown_content.encode()).decode()
#             href = f'<a href="data:text/markdown;base64,{b64_md}" download="{filename}.md" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📄 Baixar Relatório em Markdown</a>'
#             return href
            
#     except Exception as e:
#         return f'<span style="color: red;">Erro ao gerar arquivo: {str(e)}</span>'

def create_download_link(markdown_content, filename):
    """Cria um link de download para o conteúdo markdown como PDF"""
    try:
        # Criar diretório de saída
        os.makedirs("reports", exist_ok=True)

        # Caminhos dos arquivos
        md_path = f"reports/{filename}.md"
        pdf_path = f"reports/{filename}.pdf"

        # Corrigir caminhos de imagem para absolutos
        current_dir = os.getcwd().replace("\\", "/")
        processed_content = re.sub(
            r'!\[(.*?)\]\(charts/(.*?)\)',
            rf'![\1]({current_dir}/charts/\2)',
            markdown_content
        )

        # Salvar markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(processed_content)

        # Converter markdown para HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #1f77b4; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
        {markdown(processed_content)}
        </body>
        </html>
        """

        # Gerar PDF com pdfkit
        pdfkit.from_string(html_content, pdf_path)

        # Verificar se o PDF foi criado
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            b64_pdf = base64.b64encode(pdf_data).decode()
            return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}.pdf" style="background-color: #1f77b4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📄 Baixar Relatório em PDF</a>'
        else:
            # Fallback para markdown
            b64_md = base64.b64encode(markdown_content.encode()).decode()
            return f'<a href="data:text/markdown;base64,{b64_md}" download="{filename}.md" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📄 Baixar Relatório em Markdown</a>'

    except Exception as e:
        return f'<span style="color: red;">Erro ao gerar arquivo: {str(e)}</span>'

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
                    return df, separator, encoding
            except:
                continue
        
        # Se tudo falhar, tentar com parâmetros padrão
        df = pd.read_csv(file_path)
        return df, ',', 'utf-8'
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {str(e)}")
        return None, None, None

col1, col2 = st.columns([1,10])  # Ajuste a proporção conforme necessário

with col1:

    image_path = "/workspaces/EDA_MULTIAGENT_SYSTEM/eda_app/image/eistein_.jpg"  # Corrija o nome se necessário

    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            st.image(img, width=120)
        except Exception as e:
            st.error(f"Erro ao abrir a imagem: {str(e)}")
    else:
        image_path = "/mount/src/eda_multiagent_sistem/eda_app/image/eistein_.jpg"
        img = Image.open(image_path)
        st.image(img, width=120)
        #st.warning(f"Imagem não encontrada: {image_path}")

    #st.image(image, use_container_width=True)  # Ajuste o tamanho da imagem

with col2:
    st.title('Einstein Agents - Exploratory Data Analyse')  # Pode usar qualquer nível de título

st.markdown("---")

# Criar diretórios necessários
os.makedirs('charts', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Interface principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload do Arquivo")
    uploaded_file = st.file_uploader(
        'Escolha um arquivo CSV para análise', 
        type='csv',
        help="Faça upload de um arquivo CSV para análise exploratória completa"
    )

with col2:
    st.subheader("Como usar")
    st.markdown("""
    1. **Upload**: Carregue seu arquivo CSV
    2. **Pergunta**: Digite sua pergunta específica
    3. **Análise**: Clique em analisar e aguarde
    4. **Relatório**: Visualize e baixe o PDF
    """)

if uploaded_file is not None:
    # Salvar o arquivo em um local temporário
    file_path = f'data/{uploaded_file.name}'
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f'Arquivo "{uploaded_file.name}" carregado com sucesso!')
    
    # Mostrar preview dos dados
    df_preview, separator, encoding = read_csv_robust(file_path)
    
    if df_preview is not None:
        st.subheader("Preview dos Dados")
        
        # Mostrar informações sobre a detecção
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Linhas", df_preview.shape[0])
        with col_info2:
            st.metric("Colunas", df_preview.shape[1])
        with col_info3:
            st.info(f"Separador: `{separator}` | Encoding: `{encoding}`")
        
        # Mostrar preview
        st.dataframe(df_preview.head(10), use_container_width=True)
        
        # Mostrar tipos de dados
        with st.expander("Tipos de Dados Detectados"):
            types_df = pd.DataFrame({
                'Coluna': df_preview.columns,
                'Tipo': df_preview.dtypes.astype(str),
                'Valores Únicos': [df_preview[col].nunique() for col in df_preview.columns],
                'Valores Nulos': [df_preview[col].isnull().sum() for col in df_preview.columns]
            })
            st.dataframe(types_df, use_container_width=True)
    else:
        st.error("Não foi possível ler o arquivo CSV. Verifique o formato do arquivo.")

    st.markdown("---")
    
    # Pergunta do usuário
    st.subheader("Sua Pergunta")
    user_question = st.text_area(
        'Faça uma pergunta específica sobre os seus dados:',
        placeholder="Ex: Quais são as principais correlações entre as variáveis? Existem outliers significativos?",
        height=100
    )
    
    # Campo para gráfico personalizado
    st.subheader("Gráfico Personalizado (Opcional)")
    custom_chart_request = st.text_input(
        'Solicite um gráfico específico:',
        placeholder="Ex: Crie um gráfico de dispersão entre idade e salário, ou um gráfico de barras das categorias mais frequentes",
        help="O agente irá interpretar sua solicitação e gerar o código Python necessário para criar o gráfico"
    )
    
    # Combinar pergunta e solicitação de gráfico
    if custom_chart_request:
        combined_question = f"{user_question}\n\nGráfico solicitado: {custom_chart_request}"
    else:
        combined_question = user_question

    # Botão de análise centralizado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button('Iniciar Análise Completa', use_container_width=True, type="primary")
    
    if analyze_button:
        if user_question:
            # Progress bar e status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner('A equipe de agentes está analisando os dados...'):
                # Limpar gráficos antigos
                charts_dir = 'eda_app/charts'
                if os.path.exists(charts_dir):
                    for file in os.listdir(charts_dir):
                        if os.path.isfile(os.path.join(charts_dir, file)):
                            os.remove(os.path.join(charts_dir, file))

                status_text.text("Iniciando análise...")
                progress_bar.progress(20)
                
                # Criar e executar a equipe de EDA
                eda_crew = create_eda_crew(file_path, combined_question)
                
                status_text.text("Agentes trabalhando...")
                progress_bar.progress(50)
                
                result = eda_crew.kickoff()
                
                progress_bar.progress(100)
                status_text.text("Análise concluída!")
                
                # Limpar progress bar após um tempo
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

            st.markdown("---")
            
            # Cabeçalho dos resultados
            st.markdown("## Relatório de Análise Exploratória")
            
            # Extrair o conteúdo correto do resultado primeiro
            if hasattr(result, 'raw'):
                result_content = result.raw
            elif hasattr(result, 'result'):
                result_content = result.result
            else:
                result_content = str(result)
            
            # Gerar nome do arquivo baseado na data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"relatorio_eda_{timestamp}"
            
            # Botão de download
            col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
            with col_download2:
                download_link = create_download_link(result_content, filename)
                st.markdown(download_link, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Container para o relatório
            with st.container():
                st.markdown("### Relatório Completo")
                # Aplicar CSS customizado para melhor formatação
                st.markdown("""
                <style>
                .reportview-container .markdown-text-container {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                }
                .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                    color: #1f77b4;
                    border-bottom: 2px solid #e6f3ff;
                    padding-bottom: 10px;
                }
                .stMarkdown blockquote {
                    background-color: #f8f9fa;
                    border-left: 4px solid #1f77b4;
                    padding: 10px 20px;
                    margin: 20px 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Exibir o relatório completo com texto e imagens
                import re
                
                # Primeiro, exibir todo o texto do relatório
                # Remover apenas as referências de imagem para não duplicar
                text_content = re.sub(r'!\[.*?\]\(.*?\)', '', result_content)
                st.markdown(text_content, unsafe_allow_html=True)
                
                # Depois, encontrar e exibir todas as imagens mencionadas no relatório
                image_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', result_content)
                
                if image_matches:
                    st.markdown("### Gráficos e Visualizações")
                    for alt_text, image_path in image_matches:
                        if os.path.exists(image_path):
                            st.markdown(f"**{alt_text}**")
                            st.image(image_path, width='stretch')
                        else:
                            st.warning(f"Imagem não encontrada: {image_path}")

            # Seção de gráficos
            st.markdown("---")
            st.markdown("### Visualizações Geradas")
            
            charts_dir = 'charts'
            if os.path.exists(charts_dir) and os.listdir(charts_dir):
                # Organizar gráficos em colunas
                chart_files = [f for f in sorted(os.listdir(charts_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                for i, chart_file in enumerate(chart_files):
                    with st.expander(f"{chart_file.replace('_', ' ').replace('.png', '').title()}", expanded=True):
                        st.image(f'{charts_dir}/{chart_file}', width='stretch')
            else:
                st.info(" Nenhum gráfico foi gerado para esta análise.")

        else:
            st.error(' Por favor, insira uma pergunta antes de iniciar a análise.')

