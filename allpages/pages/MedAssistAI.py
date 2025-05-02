import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os

# Path to your local vector store
path = "vector_db"

# Load embeddings
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
    db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return db

# Define prompt template
def custom_prompt(my_prompt):
    prompt = PromptTemplate(template=my_prompt, input_variables=["context", "question"])
    return prompt

# Load Llama model
def load_llm(model):
    os.environ["GROQ_API_KEY"] = "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"

    llm = ChatGroq(model=model, groq_api_key=os.getenv("GROQ_API_KEY"))
    return llm

# Check if the query is medical-related
MEDICAL_KEYWORDS = [
    "symptom", "treatment", "infection", "medicine", "diagnosis", "disease",
    "allergy", "fever", "pain", "health", "prescription", "itch", "rash",
    "headache", "cough", "diabetes", "acne", "cancer", "fungal", "bacteria",
    "antibiotic", "tablet", "disorder", "virus", "covid", "flu", "injury"
]

def is_medical_query(query):
    return any(word in query.lower() for word in MEDICAL_KEYWORDS)

# Main app logic
def main():
    st.title("Ask MedAssist AI")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])    

    prompt = st.chat_input("Enter prompt")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define your medical prompt
        my_prompt = """
                Use the pieces of information provided in the context to answer the user's question.
                If you can't find the answer in the context, respectfully say that you don't have the information to answer this, don't make up an answer.
                Context: {context}
                Question: {question}
                Start the answer directly. No small talk please.
        """
        
        # Check if the query is medical-related
        if is_medical_query(prompt):
            try:
                # Load vector store and LLM
                vectorstore = load_embeddings()
                if vectorstore is None:
                    st.error("Failed to load vector store")

                # Get the chat history from session state
                chat_history = [
                        (msg["role"], msg["content"])
                        for msg in st.session_state.messages
                        if msg["role"] in ["user", "assistant"]
                    ]

                # Create the ConversationalRetrievalChain with chat history
                chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm("llama3-8b-8192"),
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                )

                # Get response from the ConversationalRetrievalChain
                response = chain.invoke({'question': prompt, 'chat_history': chat_history})
                
                # Extract the result content (ignore metadata)
                result_to_show = response
                result_to_show = response.get("answer", "Sorry, I don't have information about that.")
                # Display the result
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        else:
            # Handle general queries (greetings, small talk, etc.)
            general_prompt = f"Answer professionally : {prompt}"
            try:
                response = load_llm("llama3-8b-8192").invoke(general_prompt)
                if hasattr(response, "content"):
                    result_to_show = response.content
                else:
                    result_to_show = str(response)

                # Display the result
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()