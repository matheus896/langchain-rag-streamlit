
# RAG com Agente e Vector Store

Este código Python demonstra a implementação de um sistema de Retrieval-Augmented Generation (RAG) utilizando um agente Langchain, um Vector Store FAISS e modelos Hugging Face e Google GenAI.

## Visão Geral

O sistema RAG carrega documentos PDF, cria um índice vetorial usando embeddings Hugging Face, e utiliza um agente Langchain com ferramentas para buscar informações no índice vetorial e na web. O agente responde a perguntas com base nas informações recuperadas.

## Pré-requisitos

* Python 3.10 ou superior
* Pip

## Instalação

1. Clone este repositório:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Linux/macOS
   venv\Scripts\activate  # No Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Crie um arquivo `.env` na raiz do projeto e adicione as variáveis de ambiente necessárias. Exemplo:
   ```env
   # Variáveis de ambiente (se necessário, dependendo dos modelos LLM/Embedding que você usar)
   # GOOGLE_API_KEY= "your_google_api_key"
   # TAVILY_API_KEY= "your_tavily_api_key"
   ```

## Configuração

* **`pdf_path`**: Defina o caminho para o arquivo PDF que você deseja indexar (padrão: `"TSLA-Q3-2024-Update.pdf"`).
* **`vectorstore_path`**: Defina o caminho para salvar o banco de dados vetorial FAISS (padrão: `"vectorstore.db"`).
* **Modelos**: O código utiliza `HuggingFaceEmbeddings` para embeddings (`BAAI/bge-small-en-v1.5`) e `ChatGoogleGenerativeAI` como LLM (`gemini-2.0-flash-exp`). Você pode modificar `model_name` e `model` conforme necessário.

## Uso

1. **Execute o script Python:**
   ```bash
   python your_script_name.py # Substitua your_script_name.py pelo nome do seu arquivo python
   ```

2. **Interaja com o Agente:** O script irá:
   * Carregar e processar o documento PDF.
   * Criar ou carregar o vetorstore FAISS.
   * Inicializar um agente Langchain com ferramentas de busca vetorial e busca web.
   * Testar o agente com algumas perguntas de exemplo e imprimir as respostas.

## Funcionalidades

* **Carregamento de Documentos PDF:** Carrega e processa arquivos PDF usando `PyPDFLoader`.
* **Divisão de Texto:** Divide documentos em chunks menores usando `RecursiveCharacterTextSplitter`.
* **Embeddings:** Utiliza `HuggingFaceEmbeddings` para criar embeddings vetoriais dos chunks de texto.
* **Vector Store FAISS:** Armazena e busca embeddings de documentos usando FAISS para busca de similaridade eficiente.
* **Agente Langchain:**
    * Utiliza `ChatGoogleGenerativeAI` como o modelo de linguagem grande (LLM).
    * Emprega um prompt customizado para guiar o comportamento do agente.
    * Utiliza duas ferramentas:
        * **`VectorStoreSearch`**: Para buscar informações no vetor store FAISS.
        * **`WebSearch`**: Para realizar buscas na web usando `TavilySearchResults`.
    * Formata as interações do agente usando JSON para estruturar as ações e observações.
* **Busca na Web:** Integra a ferramenta `TavilySearchResults` para busca de informações na web quando necessário.
* **Persistência do Vector Store:** Salva e carrega o vetor store FAISS localmente para evitar a reindexação em execuções subsequentes.
* **Exemplos de Perguntas:** Inclui exemplos de perguntas para demonstrar as capacidades do agente RAG.

## Estrutura do Código

* **`load_and_process_documents(pdf_path)`**: Carrega e pré-processa documentos PDF.
* **`create_and_save_vectorstore(documents, embeddings, vectorstore_path)`**: Cria e salva o vetor store FAISS.
* **`load_vectorstore(vectorstore_path, embeddings)`**: Carrega um vetor store FAISS existente.
* **`web_search(query)`**: Realiza busca na web utilizando a ferramenta `TavilySearchResults`.
* **`VectorStoreSearchTool`**: Classe que define a ferramenta para busca no vetor store.
* **`web_search_tool_func`**: Função tool para realizar busca na web.
* **`create_agent_tools(vectorstore)`**: Cria as ferramentas para o agente Langchain.
* **`create_prompt_template(tools)`**: Define o template de prompt para o agente.
* **`create_rag_chain(prompt, llm)`**: Cria a cadeia RAG.
* **`create_agent_executor(chain, tools)`**: Cria o executor do agente.
* **`main()`**: Função principal que executa o fluxo de trabalho RAG, incluindo carregamento de documentos, criação/carregamento do vetor store, criação do agente e teste com perguntas.

## Variáveis de Ambiente

* **Variáveis de ambiente para modelos LLM e Embedding**:  Se você utilizar modelos que requerem chaves de API ou outras configurações (como modelos Google ou OpenAI), configure as variáveis de ambiente necessárias no arquivo `.env`.

## Customização

* **Modelo LLM e Embedding**: Modifique as inicializações de `HuggingFaceEmbeddings` e `ChatGoogleGenerativeAI` para usar modelos diferentes.
* **Prompt do Agente**: Ajuste `system_prompt` e `human_prompt` para alterar o comportamento e as instruções do agente.
* **Ferramentas**: Adicione ou modifique as ferramentas fornecidas ao agente em `create_agent_tools`.
* **Parâmetros de Chunking**: Ajuste `chunk_size` e `chunk_overlap` em `RecursiveCharacterTextSplitter` para otimizar o chunking de texto para seus documentos.
* **Vector Store**: Se necessário, você pode substituir FAISS por outro tipo de vector store Langchain.

## Exemplo de Uso

Após executar o script, você verá a saída do agente respondendo às perguntas de teste, indicando que o sistema RAG está funcionando. Você pode modificar a função `main()` para testar com diferentes perguntas e documentos.

**Nota:** Este README fornece uma visão geral do código e instruções básicas de uso. Para detalhes mais aprofundados, consulte os comentários no código-fonte e a documentação das bibliotecas Langchain, Hugging Face, e outras bibliotecas utilizadas.
```
