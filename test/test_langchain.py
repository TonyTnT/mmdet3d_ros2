import chromadb

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

api_key = ''
base_url= 'https://api.deepseek.com/v1'
model_name='deepseek-chat'

llm = ChatOpenAI(
     base_url=base_url,
     api_key=api_key,
     model_name=model_name)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

db_client = chromadb.HttpClient(host='localhost', port=8000)

vector_store = Chroma(
    collection_name="my_collection",
    client=db_client,
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

retriever=vector_store.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(rag_chain.invoke({"input": "what can i see"}))


question = "what can i see"
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vector_store.as_retriever(),return_source_documents=True)
print(qa_chain({"query": question}))