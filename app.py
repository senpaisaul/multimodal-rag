import streamlit as st
from rag_engine import MultimodalRAG
from graph_agent import GraphAgent
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="True Multimodal RAG",
    page_icon="📊",
    layout="wide"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
}

.main {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3, h4 {
    color: #ffffff;
}

.chat-bubble-user {
    background: #1f2933;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    color: #e5e7eb;
}

.chat-bubble-assistant {
    background: #0ea5e9;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    color: white;
}

.sidebar-box {
    background: rgba(255,255,255,0.05);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<h1>📄 True Multimodal RAG</h1>
<p style="color:#d1d5db;">
Ask questions about <b>text, images, charts</b> inside PDFs.  
Graphs are generated automatically from document data.
</p>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Capabilities")
    st.markdown("""
    - ✅ Text understanding  
    - ✅ Image & chart OCR  
    - ✅ Graph generation  
    - ❌ Hallucinations (blocked)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

if not groq_key or not pdf:
    st.info("⬅️ Please enter your Groq API key and upload a PDF.")
    st.stop()

# -------------------- Initialize Systems --------------------
rag = MultimodalRAG(groq_key)
graph = GraphAgent(groq_key)
qa_llm = ChatGroq(
    api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------- Build Vector Store --------------------
if "store" not in st.session_state:
    with st.spinner("📄 Processing document (text + images)..."):
        docs = rag.process_pdf(pdf.read())
        st.session_state.store = rag.build_store(docs)

# -------------------- Chat Input --------------------
query = st.chat_input("Ask about the document, images, or request a graph...")

if query:
    is_graph = any(k in query.lower() for k in ["plot", "graph", "chart"])
    modality = "vision" if any(k in query.lower() for k in ["image", "chart", "figure"]) else None

    docs = rag.retrieve(st.session_state.store, query, modality)

    if not docs:
        st.warning("⚠️ No relevant information found in the document.")
        st.stop()

    context = "\n".join(d.page_content for d in docs)

    # -------------------- User Message --------------------
    st.markdown(f"<div class='chat-bubble-user'><b>You:</b><br>{query}</div>",
                unsafe_allow_html=True)

    # -------------------- Graph Path --------------------
    if is_graph:
        with st.spinner("📊 Generating graph from document data..."):
            fig = graph.generate(context, query)
            st.pyplot(fig)

    # -------------------- QA Path --------------------
    else:
        system = SystemMessage(content="""
You are a document-grounded assistant.

RULES:
- Use ONLY the provided context
- Do NOT use outside knowledge
- If the answer is not present, say:
  "I cannot find this information in the document."
""")

        response = qa_llm.invoke([
            system,
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:{query}")
        ])

        st.markdown(
            f"<div class='chat-bubble-assistant'><b>Assistant:</b><br>{response.content}</div>",
            unsafe_allow_html=True
        )
