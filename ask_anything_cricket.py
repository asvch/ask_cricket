# Import Langchain dependencies
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Bring in streamlit for UI dev
import streamlit as st
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create LLM using Langchain
os.environ["GOOGLE_API_KEY"] = "*********"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

st.set_page_config(
    page_title="Ask Anything Cricket!",
    page_icon="🏏"
)

# Setup the app title
st.title('Ask Anything Cricket!')


system_prompt = (
        "Only use the given context to answer the question. "
        "Convey the answer so it is easy to understand for a beginner. "
        "If you don't know the answer, say you don't know. "
        "Do not answer with any information outside of the given context. "
        "Context: {context}"
    )

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
question = st.chat_input('Ask your questions here')

# Function to load PDF
@st.cache_resource    # wrapper so streamlit doesn't need to reload it each time, makes it faster
def load_pdf():
    
    DATA_PATH = "docs"
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")  # threshold type a value that can be experimented with for better results

    docs = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(docs, embeddings , persist_directory="cricket_db")

    # Retun the vector database
    return vectorstore

index = load_pdf()

# Create a Q&A chain
chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = index.as_retriever())

if question:
        # Display the prompt
        st.chat_message('user').markdown(question)

        # Store the user prompt in state
        st.session_state.messages.append({'role': 'user', 'content': question})

        # Send prompt to PDF Q&A Chain
        response = chain.invoke(question)

        # Show LLM Response
        st.chat_message(system_prompt).markdown(response['result'])

        # Store LLM response in state
        st.session_state.messages.append({'role': system_prompt, 'content': response['result']})
