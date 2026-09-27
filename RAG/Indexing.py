import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic import hub

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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' 

#indexing
loader = PyPDFLoader('data/unstructured/cost_of_land.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents = all_splits,
    embedding = embeddings,
    persist_directory = "./housing_data_db"
)

#retrieving
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#generating
llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)

print("English RAG System ready for testing!")

question = "What determines the land lease price of urban land?"
rag_chain.invoke(question)
