#!/usr/bin/env python3
"""
Script para executar o Sistema EDA com Agentes CrewAI
"""
import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Verifica se o arquivo .env existe e está configurado"""
    if not os.path.exists('.env'):
        print("Arquivo .env não encontrado!")
        print("Execute primeiro: python install.py")
        return False
    
    # Verificar se a chave está configurada
    with open('.env', 'r') as f:
        content = f.read()
        if 'sua_chave_openai_aqui' in content:
            print("Chave da OpenAI não configurada no arquivo .env")
            print("Edite o arquivo .env e substitua 'sua_chave_openai_aqui' pela sua chave real")
            return False
    
    print("Arquivo .env configurado")
    return True

def check_directories():
    """Verifica se os diretórios necessários existem"""
    directories = ['charts', 'data', 'reports']
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Criando diretório '{directory}'...")
            os.makedirs(directory, exist_ok=True)
    
    print("Diretórios verificados")
    return True

def load_env_file():
    """Carrega as variáveis do arquivo .env"""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def run_streamlit():
    """Executa a aplicação Streamlit"""
    print("\nIniciando aplicação...")
    print("A aplicação será aberta em: http://localhost:8501")
    print("Para parar a aplicação, pressione Ctrl+C")
    print("-" * 50)
    
    try:
        # Carregar variáveis de ambiente
        load_env_file()
        
        # Executar Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n Aplicação encerrada pelo usuário")
    except subprocess.CalledProcessError as e:
        print(f"\n Erro ao executar a aplicação: {e}")
        print("\nVerifique se todas as dependências estão instaladas:")
        print("python install.py")
    except FileNotFoundError:
        print("\n Streamlit não encontrado!")
        print("Execute: python install.py")

def main():
    """Função principal"""
    print(" Sistema EDA com Agentes CrewAI")
    print("=" * 40)
    
    # Verificar arquivo .env
    if not check_env_file():
        return False
    
    # Verificar diretórios
    check_directories()
    
    # Executar aplicação
    run_streamlit()
    
    return True

if __name__ == "__main__":
    main()
