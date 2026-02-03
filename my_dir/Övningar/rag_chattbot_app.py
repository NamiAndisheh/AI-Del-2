import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
import io
import os

# API-SETUP

# API-nyckel (bör användas från miljövariabel i produktion)
API_KEY = os.getenv("API_KEY")

@st.cache_resource
def get_client():
# Skapar och cachar Gemini-klienten
    return genai.Client(api_key=API_KEY)

client = get_client()

# RAG-FUNKTIONER

def create_embeddings(text, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    # Omvandlar text till siffror
    return client.models.embed_content(
        model=model, 
        contents=text, 
        config=types.EmbedContentConfig(task_type=task_type)
    )

def cosine_similarity(vec1, vec2):
    # Beräknar cosinus-likhet mellan två vektorer
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, embeddings, k=5):
    # Hittar de mest relevanta chunks baserat på en fråga
    query_embedding = create_embeddings(query).embeddings[0].values
    similarity_scores = []
    
    for i, chunk_embedding in enumerate(embeddings.embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, similarity_score))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    
    return [chunks[index] for index in top_indices]

def generate_user_prompt(query, chunks, embeddings):
    # Skapar en prompt med kontext från semantisk sökning
    context = "\n".join(semantic_search(query, chunks, embeddings))
    return f"Frågan är: {query}\n\nHär är kontexten:\n{context}"

def generate_response(query, chunks, embeddings):
    # Genererar ett svar baserat på RAG
    system_prompt = """Du är en hjälpsam assistent som svarar på frågor baserat endast på den kontext som ges.
Om svaret inte finns i kontexten, säg "Det vet jag inte baserat på dokumentet".
Svara enkelt, tydligt och på svenska.
Formulera dig enkelt och dela upp svaret i fina stycken."""
    
    user_prompt = generate_user_prompt(query, chunks, embeddings)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=genai.types.GenerateContentConfig(system_instruction=system_prompt),
        contents=user_prompt
    )
    return response.text

# PDF-PROCESSING

def process_pdf(uploaded_file):
    # Läser PDF, chunkar texten och skapar embeddings
    # Läs PDF från uppladdad fil
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    
    # Extrahera all text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Chunking
    chunks = []
    n = 1000  # Chunk-storlek
    overlap = 200  # Överlappning
    
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    
    # Skapa embeddings
    embeddings = create_embeddings(chunks)
    
    return text, chunks, embeddings

# STREAMLIT UI

# Sidkonfiguration
st.set_page_config(
    page_title="RAG Chattbot",
    layout="wide"
)

# Titel och beskrivning
st.title("Nami AI")
st.write("Ladda upp en PDF-fil och ställ frågor")

# Initiera session state
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_length" not in st.session_state:
    st.session_state.document_length = 0

# Sidebar för PDF-uppladdning
with st.sidebar:
    st.header("Ladda upp dokument")
    
    uploaded_file = st.file_uploader(
        "Välj en PDF-fil",
        type=['pdf'],
        help="Ladda upp en PDF-fil som chattboten ska kunna svara på frågor om."
    )
    
    if uploaded_file is not None:
        if st.button("Bearbeta PDF", use_container_width=True):
            with st.spinner("Bearbetar PDF-filen..."):
                try:
                    text, chunks, embeddings = process_pdf(uploaded_file)
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.pdf_processed = True
                    st.session_state.document_length = len(text)
                    st.session_state.chat_history = []  # Rensa chatthistorik
                    st.success("PDF bearbetad!")
                except Exception as e:
                    st.error(f"Fel vid bearbetning: {str(e)}")
    
    # Visa status
    if st.session_state.pdf_processed:
        st.divider()
        st.subheader("Dokumentinfo")
        st.write(f"**Antal chunks:** {len(st.session_state.chunks)}")
        st.write(f"**Dokumentlängd:** {st.session_state.document_length:,} tecken")
        
        if st.button("Rensa chatthistorik", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("Ladda upp ny PDF", use_container_width=True):
            st.session_state.chunks = None
            st.session_state.embeddings = None
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.document_length = 0
            st.rerun()

# Huvudområde för chatt
if not st.session_state.pdf_processed:
    st.info("Ladda upp en PDF-fil i sidofältet för att börja chatta!")
    
else:
    # Visa chatthistorik
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chattinput
    if question := st.chat_input("Ställ en fråga om dokumentet..."):
        # Lägg till användarens fråga i historiken
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Visa användarens meddelande
        with st.chat_message("user"):
            st.write(question)
        
        # Generera svar
        with st.chat_message("assistant"):
            with st.spinner("Tänker..."):
                try:
                    answer = generate_response(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings
                    )
                    st.write(answer)
                    
                    # Lägg till svaret i historiken
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Fel vid generering av svar: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})



