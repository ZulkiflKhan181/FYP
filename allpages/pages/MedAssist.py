from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
import os


# Setting up API key 
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

model="llama3-8b-8192"

# Defining the chat model

def load_llm(model):
 llm = ChatGroq(model=model,
                groq_api_key=GROQ_API_KEY)
 return llm

# Loading the PDF
pdf_path = "C:\\Users\\Gamer Tech\\Desktop\\Medical Info.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Splitting text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Converting text chunks into embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  


vector_store = FAISS.from_documents(docs, embeddings)

# Saving the FAISS vector 
vector_store.save_local("vector_database")

# Creating Chain
my_prompt="""
Use the peices of information provided in the context to asnwer user's question.
If you cant find the answer in the context, respectfully say that you dont have the information to ansswer this, dont make up an answer.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
def custom_prompt(my_prompt):
  prompt=PromptTemplate(template=my_prompt, input_variables= ["context", "question"])
  return prompt

path="vector_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
db=FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

qa_chain=RetrievalQA.from_chain_type(
  llm=load_llm(model),
  chain_type="stuff",
  retriever=db.as_retriever(search_kwargs={"k":3}),
  chain_type_kwargs={"prompt":custom_prompt(my_prompt)}
)

# Invoking
user_query = input("Search")
response = qa_chain.invoke({'query':user_query})
print(response['result'])



