from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import RePhraseQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create LLM using Langchain
llm = ChatOpenAI()

st.set_page_config(
    page_title="Ask About Cricket!",
    page_icon="🏏"
)

# Setup the app title
st.title('Ask About Cricket!')


system_prompt = (
        "Use ONLY the given context to answer the question. "
        "Convey the answer so it is easy to understand for a beginner. "
        "If you cannot answer the question based on the given context, say you do not know. "
        "Do not answer with any information outside of the given context. "
        "Context: {context}"
    )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
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
@st.cache_resource    # wrapper so streamlit doesn't need to reload it each time, makes it all faster
def load_pdf():
    
    DATA_PATH = "data"
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")  # threshold type a value that can be experimented with for better results

    docs = text_splitter.split_documents(docs)

    # if there's an error with Chroma.from_documents, remove some docs from data folder and try again
    vectorstore = Chroma.from_documents(docs, embeddings , persist_directory="chromadb_cricket")

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# Advanced RAG - Query Rewriting and Prompt Compression
    # Query Rewriting using LLM
    retriever_from_llm = RePhraseQueryRetriever.from_llm(   # using llm to rephrase query & make it more compact
        retriever=retriever, llm=llm
    )

    # Prompt Compression (Post-Retrieval Stage)
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.50)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever_from_llm
    )

    return compression_retriever

compression_retriever = load_pdf()

question_answer_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(compression_retriever, question_answer_chain)

if question:
        # Display the prompt
        st.chat_message('user').markdown(question)

        # Store the user prompt in state
        st.session_state.messages.append({'role': 'user', 'content': question})

        # Send prompt to PDF Q&A Chain
        # response = chain.invoke(question)
        response = chain.invoke({"input": question})

        # Show LLM Response
        st.chat_message(system_prompt).markdown(response['answer'])

        # Store LLM response in state
        st.session_state.messages.append({'role': system_prompt, 'content': response['answer']})