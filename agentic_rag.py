import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.language_models import BaseLanguageModel

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Definir o caminho para o arquivo PDF
pdf_path = "TSLA-Q3-2024-Update.pdf"  # Substitua pelo caminho do seu PDF

# Definir o caminho para o banco de dados vetorial
vectorstore_path = "vectorstore.db"

# Inicializar o modelo de embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True}
)

# Carregar o modelo de linguagem
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Função para carregar e processar os documentos
def load_and_process_documents(pdf_path): # Removido embeddings daqui, não é usado neste passo
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    return documents

# Função para criar e salvar o vetorstore
def create_and_save_vectorstore(documents, embeddings, vectorstore_path):
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

# Função para carregar o vetorstore
def load_vectorstore(vectorstore_path, embeddings):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Inicializar a ferramenta de busca na web
web_search_tool = TavilySearchResults(k=10)

# Função para busca na web
def web_search(query: str):
    return web_search_tool.run(query)

class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: ChatGoogleGenerativeAI(temperature=0))

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

# Criar tool para busca na web
@tool
def web_search_tool_func(query: str) -> str:
    """Tool for performing web search."""
    return web_search(query)

class VectorStoreSearchTool(BaseVectorStoreTool, BaseTool):
    """Tool for searching the vector store."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name}. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        from langchain.chains.retrieval_qa.base import RetrievalQA
        chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return chain.invoke( # Changed from chain.ainvoke to chain.invoke
            {chain.input_key: query}
            # Removed config and run_manager completely
        )[chain.output_key]


# Criar as ferramentas para o agente
def create_agent_tools(vectorstore):
    tools = [
        VectorStoreSearchTool(
            name="VectorStoreSearch",
            description=VectorStoreSearchTool.get_description(
                "the vector store", "the information in the vector store"
            ),
            vectorstore=vectorstore,
            llm=llm,
        ),
        Tool(
            name="WebSearch",
            func=web_search_tool_func,
            description="Use this to perform a web search for information."
        ),
    ]
    return tools

# Criar o prompt do sistema
system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
Always try the "VectorStoreSearch" tool first. Only use "WebSearch" if the vector store does not contain the required information.
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:

{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}

Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
content_copy
download
Use code with caution.

$JSON_BLOB

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:

{{
"action": "Final Answer",
"action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

# Criar prompt do usuário
human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""

# Função para criar o template do prompt
def create_prompt_template(tools):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    prompt = prompt.partial(
        tools=render_text_description_and_args(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    return prompt

# Criar a cadeia RAG
def create_rag_chain(prompt, llm):
    chain = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )
    return chain

# Função para criar o agente
def create_agent_executor(chain, tools):
    agent_executor = AgentExecutor(
        agent=chain,
        tools=tools,
        handle_parsing_errors=False,
        verbose=True,
    )
    return agent_executor

# Execução principal
def main():

    # Carregar documentos
    documents = load_and_process_documents(pdf_path) # Removido embeddings daqui, não é usado neste passo

    # Verificar se o vetorstore já existe, se não, cria e salva
    if not os.path.exists(vectorstore_path):
        vectorstore = create_and_save_vectorstore(documents, embeddings, vectorstore_path)
        print("Vetorstore criado e salvo com sucesso!")
    else:
        vectorstore = load_vectorstore(vectorstore_path, embeddings)
        print("Vetorstore carregado com sucesso!")

    # Criar ferramentas para o agente
    tools = create_agent_tools(vectorstore)

    # Criar o template do prompt
    prompt = create_prompt_template(tools)

    # Criar a cadeia RAG
    chain = create_rag_chain(prompt, llm)

    # Criar o executor do agente
    agent_executor = create_agent_executor(chain, tools)
    return agent_executor

if __name__ == "__main__":
    (main())