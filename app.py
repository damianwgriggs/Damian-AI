import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
from openai import OpenAI
import logging
from database import ARTICLES 

# --- 0. Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Load the Constitution ---
try:
    with open("damian_constitution.md", "r", encoding='utf-8') as f:
        CONSTITUTION = f.read()
    logging.info("Successfully loaded the Damian AI Constitution.")
except FileNotFoundError:
    st.error("FATAL ERROR: `damian_constitution.md` not found. Please ensure it's in the same folder.")
    st.stop()

# --- 2. Local AI Model Setup (Deployment Ready) ---
@st.cache_resource
def get_local_client():
    """Initializes and caches the OpenAI client for the server."""
    try:
        # --- MODIFICATION: Securely load URL from Hugging Face Secrets ---
        llm_endpoint = os.environ.get("LLM_ENDPOINT_URL", "http://127.0.0.1:1234/v1")
        
        client = OpenAI(base_url=llm_endpoint, api_key="not-needed")
        client.models.list()
        logging.info(f"Successfully connected to LLM at {llm_endpoint}")
        return client
    except Exception as e:
        st.error(f"Failed to connect to the AI model. Is the server running and is the endpoint URL correct? Error: {e}")
        st.stop()

# --- File paths for persistence (less critical on ephemeral HF Spaces) ---
FAISS_INDEX_PATH = "faiss_index.bin"
DB_CONTENT_PATH = "db_content.json"

# --- 3. Knowledge Base (Vector DB) Components ---
@st.cache_resource
def get_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- MODIFICATION: This function now runs silently on startup ---
@st.cache_resource
def build_vector_database():
    """Builds and caches the vector database from database.py."""
    logging.info("Building vector database from source...")
    model = get_sentence_transformer_model()
    all_chunks, chunk_metadata = [], []
    
    for article in ARTICLES:
        text, source = article.get("content", ""), article.get("source", "Unknown Source")
        if text:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip().split()) > 10]
            for para in paragraphs:
                all_chunks.append(para)
                chunk_metadata.append({'source_url': source})

    if not all_chunks:
        logging.error("`database.py` is empty. Cannot build knowledge base.")
        return None, None
        
    logging.info(f"Generated {len(all_chunks)} chunks. Creating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
    index.add_with_ids(embeddings.astype('float32'), np.arange(len(all_chunks)).astype('int64'))
    
    db_content = {"chunks": all_chunks, "metadata": chunk_metadata}
    logging.info("Vector database built successfully.")
    return index, db_content

# --- 4. Cognitive Engine (ChatManager) ---
class ChatManager:
    # ... (No changes needed inside the ChatManager class) ...
    def __init__(self, client, model_name, memory_token_limit=3000, recent_message_count=10):
        self.client = client
        self.model_name = model_name
        self.memory_token_limit = memory_token_limit
        self.recent_message_count = recent_message_count

    def _count_tokens(self, text): return len(text) // 4
    
    def _summarize_conversation(self, history):
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        prompt = f"Summarize the key points of this conversation transcript concisely:\n\n{history_text}\n\nCONCISE SUMMARY:"
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2)
            return response.choices[0].message.content.strip()
        except Exception: return "Summary failed."

    def _retrieve_relevant_knowledge(self, query, index, db_content, top_k=5):
        model = get_sentence_transformer_model()
        query_embedding = model.encode([query])
        _, indices = index.search(query_embedding.astype('float32'), top_k)
        retrieved_chunks = [db_content["chunks"][i] for i in indices[0] if i != -1]
        retrieved_sources = {db_content["metadata"][i]['source_url'] for i in indices[0] if i != -1}
        context = "\n\n---\n\n".join(retrieved_chunks)
        return context, list(retrieved_sources)

    def decide_strategy(self, user_query):
        prompt = f"Analyze the user's query. Does it contain multiple distinct questions requiring a synthesized answer, or is it a single, focused question?\n\nUser Query: \"{user_query}\"\n\nRespond with ONLY ONE of the following words:\n- `DirectAnswer`\n- `Synthesis`"
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0)
            strategy = response.choices[0].message.content.strip().replace("`", "")
            if strategy not in ['DirectAnswer', 'Synthesis']: return 'DirectAnswer'
            return strategy
        except Exception: return 'DirectAnswer'

    def execute_direct_answer(self, user_query, context, sources, history, long_term_memory):
        prompt = f"{CONSTITUTION}\n\n**Supporting Context:**\n- Retrieved Knowledge: {context}\n- Sources: {sources}\n- Conversation Summary: {long_term_memory}\n\n**User Query:** {user_query}"
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)
        return response.choices[0].message.content.strip()

    def execute_synthesis(self, user_query, context, sources, history, long_term_memory):
        prompt = f"{CONSTITUTION}\n\nYour task is to identify the single, underlying principle connecting all of the user's inquiries and write a single, cohesive analysis.\n\n**Supporting Context:**\n- Retrieved Knowledge: {context}\n- Sources: {sources}\n- Conversation Summary: {long_term_memory}\n\n**User Inquiries to Synthesize:**\n{user_query}"
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)
        return response.choices[0].message.content.strip()

    def get_response(self, user_query, vector_index, db_content):
        history, long_term_memory = st.session_state.messages, st.session_state.long_term_memory
        if self._count_tokens(json.dumps(history)) > self.memory_token_limit:
            history_to_summarize, short_term_history = history[:-self.recent_message_count], history[-self.recent_message_count:]
            new_summary = self._summarize_conversation(history_to_summarize)
            st.session_state.long_term_memory, long_term_memory = new_summary, new_summary
            st.session_state.messages = short_term_history
            st.toast("PGE: Long-term conversational memory updated!", icon="🧠")
        context, sources = self._retrieve_relevant_knowledge(user_query, vector_index, db_content)
        st.session_state.sources = sources
        strategy = self.decide_strategy(user_query)
        try:
            if strategy == 'Synthesis':
                return self.execute_synthesis(user_query, context, sources, history, long_term_memory)
            else:
                return self.execute_direct_answer(user_query, context, sources, history, long_term_memory)
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return "A critical error occurred in my cognitive process."

# --- 5. Streamlit User Interface ---
st.set_page_config(page_title="Damian AI", layout="wide")
st.title("Damian AI")
st.caption("Cognitive Architecture Edition")

# --- MODIFICATION: Automated knowledge base loading ---
if 'vector_index' not in st.session_state:
    with st.spinner("Initializing knowledge base... This may take a moment on first start."):
        st.session_state.vector_index, st.session_state.db_content = build_vector_database()

if "messages" not in st.session_state: st.session_state.messages = []
if 'sources' not in st.session_state: st.session_state.sources = []
if 'long_term_memory' not in st.session_state: st.session_state.long_term_memory = "No conversation history."

with st.sidebar:
    st.header("⚙️ System Configuration")
    model_name = st.text_input("Local Model Name", "local-model")
    st.info("Ensure this matches the model loaded in your local server.")
    st.header("🧠 PGE Memory")
    memory_token_limit = st.slider("Memory Trigger (Tokens)", 1000, 8000, 3000, 500)
    recent_message_count = st.slider("Recent Events", 2, 20, 10, 2)
    st.header("📚 Knowledge Base")
    # --- MODIFICATION: UI simplified ---
    if st.session_state.vector_index is not None:
        st.success(f"Knowledge base loaded successfully.")
    else:
        st.error("Knowledge base failed to load.")
    st.header("🔗 Last Response Sources")
    st.json(st.session_state.get('sources', []), expanded=False)

client = get_local_client()
chat_manager = ChatManager(client, model_name, memory_token_limit, recent_message_count)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question to Damian AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    if st.session_state.vector_index is None:
        st.warning("Knowledge base is not loaded. The app may not function correctly.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_manager.get_response(prompt, st.session_state.vector_index, st.session_state.db_content)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

