import streamlit as st
from agentic_rag import main as rag_main  # Importa a função principal do código RAG

# Configuração da página Streamlit
st.set_page_config(
    page_title="RAG System Interface",
    page_icon="📚",
    layout="wide"
)

# Título da aplicação
st.title("📚 Sistema RAG: Recuperação e Geração Aumentada")
st.markdown("""
Este aplicativo permite que você faça perguntas sobre documentos processados e obtenha respostas com base em um sistema RAG (Retrieval-Augmented Generation).
""")

# Função para executar o RAG de forma síncrona
def run_rag(query):
    """
    Executa o sistema RAG com a consulta fornecida.
    Retorna a resposta gerada pelo sistema.
    """
    # Aqui, chamamos a função principal do código RAG
    agent_executor = rag_main()
    response = agent_executor.invoke({"input": query})
    return response['output']

# Interface do usuário
st.sidebar.header("🔍 Consulta ao Sistema RAG")
query = st.sidebar.text_input("Digite sua pergunta aqui:", placeholder="Ex: Qual foi o custo total de receitas no Q3 de 2024?")

def main():
    if query:
        with st.spinner("Processando sua pergunta..."):
            # Executa o sistema RAG
            response = run_rag(query)

        # Exibe a resposta
        st.subheader("📝 Resposta:")
        st.write(response)

if __name__ == "__main__":
    main()

# Explicação sobre o sistema
st.sidebar.markdown("""
### 📚 Sobre o Sistema
- **Recuperação**: O sistema busca informações relevantes no banco de dados vetorial local.
- **Geração**: Se necessário, complementa a busca com informações da web.
- **Modelos**: Usa modelos avançados como Google Generative AI e HuggingFace.
""")

# Rodapé
st.sidebar.markdown("""
---
Desenvolvido com ❤️ usando [Streamlit](https://streamlit.io/) e [RAG](https://github.com/HKUDS/LightRAG).
""")