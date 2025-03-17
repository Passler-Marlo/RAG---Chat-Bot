from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.schema import BaseRetriever, Document
from pydantic import Field
from typing import List
import streamlit as st
import subprocess
import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain



# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Chatbot Settings")

# Temperature Slider
temperature = st.sidebar.slider("Adjust chatbot creativity (temperature)", 0.0, 1.0, 0.01)

# Toggle for "Hiss Mode"
hiss_mode = st.sidebar.toggle("Enable Hiss Mode 🐍")

# Response Style Selector
response_style = st.sidebar.selectbox("Choose response style:", ["Neutral", "Analytic", "Enthusiastic"])

# Clear History Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

# ---------- LLM Setup ----------
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
os.environ["HUGGING_TOKEN"] = st.secrets["HUGGING_TOKEN"]

hf_token = os.environ("HUGGING_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id=hf_model,
    task='text-generation',
    temperature=temperature,
    top_p=0.95,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_token
)

# ---------- Embeddings & Vector Database ----------
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
e_vector_db = FAISS.load_local("content/e_faiss_index", embeddings, allow_dangerous_deserialization=True)
q_vector_db = FAISS.load_local("content/q_faiss_index", embeddings, allow_dangerous_deserialization=True)

# ---------- Combined Retriever ----------
class CombinedRetriever(BaseRetriever):
    retrievers: List[BaseRetriever] = Field(default_factory=list)
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = []
        seen_docs = set()
        for retriever in self.retrievers:
            # Use _get_relevant_documents if available, passing the run_manager argument
            if hasattr(retriever, "_get_relevant_documents"):
                retrieved = retriever._get_relevant_documents(query, run_manager=run_manager)
            else:
                # Fallback to get_relevant_documents, also passing run_manager if supported
                retrieved = retriever.get_relevant_documents(query, run_manager=run_manager)
            for doc in retrieved:
                if doc.page_content not in seen_docs:
                    seen_docs.add(doc.page_content)
                    docs.append(doc)
        return docs


e_retriever = e_vector_db.as_retriever(search_kwargs={"k": 1}, search_type="similarity")
q_retriever = q_vector_db.as_retriever(search_kwargs={"k": 2}, search_type="similarity")
combined_retriever = CombinedRetriever(retrievers=[e_retriever, q_retriever])

# ---------- Prompt Templates with Explicit Context ----------
neutral_template = """
You are an expert Python assistant that only answers questions strictly related to Python programming. You simplify complex Python concepts with clear explanations and real-world examples.

**IMPORTANT:** ONLY answer questions about Python programming. If the question is not related to Python programming, immediately reply with:
"I am a simple Python and only know Python!" and do not provide any additional information.

The following context is provided for your internal processing only and must NOT be included or referenced in your final answer:
{context}

Now, based solely on this internal context, answer the following question. Do not mention or reveal any internal examples or context details.

Question: {input}
Answer:"""

analytic_template = """
You are an expert Python assistant that always includes coding examples in your responses. You break down complex Python concepts step by step and always support your explanations with example code snippets.

**IMPORTANT:** ONLY answer questions about Python programming. If the question is not related to Python programming, immediately reply with:
"If it's not about Python code I am not interested!" and do not provide any additional information.

The following context is provided for your internal processing only and must NOT be included or referenced in your final answer:
{context}

Now, based solely on this internal context, answer the following question. Do not mention or reveal any internal examples or context details.

Question: {input}
Answer:"""

enthusiastic_template = """
You are an enthusiastic Python assistant who believes Python is the best programming language on Earth! You use exuberant, energetic language and lots of exclamation points to celebrate Python’s elegance and capabilities.

**IMPORTANT:** ONLY answer questions about Python programming. If the question is not related to Python programming, immediately reply with:
"Why waste time on lesser topics when we could keep on talking about Python?!!" and do not provide any additional information.

The following context is provided for your internal processing only and must NOT be included or referenced in your final answer:
{context}

Now, based solely on this internal context, answer the following question. Do not mention or reveal any internal examples or context details.

Question: {input}
Answer:"""

if response_style == "Analytic":
    template = analytic_template
elif response_style == "Enthusiastic":
    template = enthusiastic_template
else:
    template = neutral_template

# Add an empty assistant turn to enforce a single final answer.
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ---------- Initialize Bot (No Caching) ----------
def init_bot():
    doc_retriever = create_history_aware_retriever(llm, combined_retriever, prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(doc_retriever, doc_chain)

rag_bot = init_bot()

# ---------- Main Chat Interface ----------
st.title("The Chatty Python")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Curious minds wanted!"):
    st.chat_message("human").markdown(prompt_input)
    st.session_state.messages.append({"role": "human", "content": prompt_input})
    
    with st.spinner("The python is thinking..."):
        # Hiss mode takes precedence.
        if hiss_mode:
            response = "Hisssssss..."
        else:
            # Retrieve relevant documents and build context from metadata.
            docs = combined_retriever.get_relevant_documents(prompt_input)
            context_text = "\n\n".join(
                [doc.metadata.get("example") or doc.metadata.get("answer") or "" for doc in docs]
            )
            try:
                result = rag_bot.invoke({
                    "input": prompt_input,
                    "chat_history": st.session_state.messages,
                    "context": context_text
                })
                response = result["answer"]
            except Exception as e:
                response = f"An error occurred: {e}"
                
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
