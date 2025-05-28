import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os

# Streamlit UI
st.title("Chat with Your PDFs (RAG + History)")
api_key = st.text_input("Groq API Key", type="password")
session_id = st.text_input("Session ID", value="default")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_history(session) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if files:
        all_docs = []
        for i, file in enumerate(files):
            path = f"temp_{i}.pdf"
            with open(path, "wb") as f:
                f.write(file.read())
            all_docs.extend(PyPDFLoader(path).load())

        # Split + embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        chunks = splitter.split_documents(all_docs)
        vs = Chroma.from_documents(chunks, embedding=embeddings)
        retriever = vs.as_retriever()

        # Prompts
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase if needed for standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer with the following context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware = create_history_aware_retriever(llm, retriever, history_prompt)
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware, qa_chain)

        chat_chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        query = st.text_input("Ask something:")
        if query:
            history = get_history(session_id)
            result = chat_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
            st.write("**Assistant:**", result["answer"])

            with st.expander("Chat History"):
                for msg in history.messages:
                    st.markdown(f"**{msg.type.title()}**: {msg.content}")
else:
    st.warning("Enter your Groq API Key to begin.")
