import os
from pathlib import Path

# Ladda API_KEY från .env i my_dir 
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            if line.strip().startswith("API_KEY="):
                os.environ["API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY saknas. Skapa .env i projektroten med raden: API_KEY=din_nyckel")

import streamlit as st
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from PIL import Image
import io
import time

# STANDARD KOKBOK - Laddas automatiskt vid start
DEFAULT_PDF_PATH = "/Users/namiandisheh/Desktop/AI Del 2/USU-Student-Cookbook-FINAL-1.pdf"

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

client = get_client()

# RETRY-FUNKTION (används pågrund av många anrop till API)

def retry_api_call(func, max_retries=3):
    # Använder exponential backoff (väntar längre för varje försök)
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            # Kolla om det är ett 429-fel (för det är vad för fel som uppstår)
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                # Om det inte är 429-fel eller sista försöket, kasta felet
                raise e

# RAG-FUNKTIONER

def create_embeddings(text, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    # Omvandlar text till siffror med retry-logik
    # Om text är en lista, dela upp i batchar om max 100
    if isinstance(text, list) and len(text) > 100:
        all_embeddings = []
        for i in range(0, len(text), 100):
            batch = text[i:i+100]
            
            # Använder retry_api för att kunna anropa API:et via retry‑funktionen
            def api_call(b=batch):
                return client.models.embed_content(
                    model=model, 
                    contents=b, 
                    config=types.EmbedContentConfig(task_type=task_type)
                )
            
            result = retry_api_call(api_call)
            all_embeddings.extend(result.embeddings)
            
            # Vänta mellan batchar för att undvika rate limiting
            time.sleep(0.5)
        
        # Skapa ett objekt som liknar det vanliga svaret
        class BatchedEmbeddings:
            def __init__(self, embeddings):
                self.embeddings = embeddings
        
        return BatchedEmbeddings(all_embeddings)
    else:
        def api_call():
            return client.models.embed_content(
                model=model, 
                contents=text, 
                config=types.EmbedContentConfig(task_type=task_type)
            )
        
        return retry_api_call(api_call)

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

# VISION-FUNKTION (Bild -> Ingredienser)

def identify_ingredients(image):
    # Identifierar ingredienser i en bild med retry-logik
    prompt = """Titta på bilden. Lista alla matvaror och ingredienser du ser. 
    Svara bara med en enkel lista separerad med kommatecken, t.ex: 'mjölk, ägg, morot'.
    Om du inte kan identifiera några ingredienser, svara med 'Inga ingredienser hittades'."""
    
    def api_call():
        return client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
    
    response = retry_api_call(api_call)
    return response.text

def generate_recipe(ingredients, context_chunks):
    # Genererar ett recept baserat på ingredienser och kokbokskontext
    # Receptet får ENDAST använda bas-ingredienserna från bilden
    context = "\n".join(context_chunks)
    
    system_prompt = """Du är en kock som skapar recept.

STRIKTA REGLER:
1. Receptet får ENDAST använda ingredienser från listan 'Bas-ingredienser'.
2. INGA andra ingredienser får förekomma i ingredienslistan eller i receptstegen.
3. Om du vill ge extra tips, lista dem under 'Valfria tillägg' - de får INTE användas i receptet.
4. Använd kokboken endast som inspiration för tillagningsmetod, inte för nya ingredienser.
5. Om bas-ingredienserna inte räcker för ett recept, säg det tydligt.
6. Svara på svenska."""
    
    user_prompt = f"""Bas-ingredienser (från bilden - ENDAST dessa får användas i receptet):
{ingredients}

Kokbokskontext (använd endast som inspiration för tillagning, INTE för nya ingredienser):
{context}

Svara i detta format:

**Ingredienser (endast från bilden):**
- ...

**Recept:**
1. ...

**Valfria tillägg (ej i receptet ovan, endast förslag):**
- ..."""
    
    def api_call():
        return client.models.generate_content(
            model="gemini-2.0-flash",
            config=genai.types.GenerateContentConfig(system_instruction=system_prompt),
            contents=user_prompt
        )
    
    response = retry_api_call(api_call)
    return response.text

# PDF-PROCESSING

def process_pdf(uploaded_file):
    # Läser PDF från uppladdad fil, chunkar texten och skapar embeddings
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    
    # Extrahera all text - med felhantering för varje sida
    text = ""
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        except Exception:
            # Hoppa över sidor som inte kan läsas
            continue
    
    if not text.strip():
        raise ValueError("Kunde inte extrahera text från PDF:en")
    
    # Chunking
    chunks = []
    n = 1000  # Chunk-storlek
    overlap = 200  # Överlappning
    
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    
    # Skapa embeddings
    embeddings = create_embeddings(chunks)
    
    return text, chunks, embeddings

def process_pdf_from_path(file_path):
    pdf_reader = PdfReader(file_path)
    
    # Extrahera all text - med felhantering för varje sida
    text = ""
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        except Exception:
            # Hoppa över sidor som inte kan läsas
            continue
    
    if not text.strip():
        raise ValueError("Kunde inte extrahera text från PDF:en")
    
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
st.set_page_config(
    page_title="Kylskåps-Kocken",
    layout="wide"
)

# Titel och beskrivning
st.title("AI-Kocken")
st.write("Fota ditt kylskåp, ladda upp en kokbok, och få ett recept!")

# Initiera session state
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "document_length" not in st.session_state:
    st.session_state.document_length = 0
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "identified_ingredients" not in st.session_state:
    st.session_state.identified_ingredients = None
if "generated_recipe" not in st.session_state:
    st.session_state.generated_recipe = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

# AUTO-LADDA STANDARD KOKBOK vid första körning
if not st.session_state.pdf_processed and os.path.exists(DEFAULT_PDF_PATH):
    with st.spinner("Laddar standard kokbok (USU Student Cookbook)..."):
        try:
            text, chunks, embeddings = process_pdf_from_path(DEFAULT_PDF_PATH)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.pdf_processed = True
            st.session_state.document_length = len(text)
            st.session_state.current_pdf_name = "USU Student Cookbook"
        except Exception as e:
            st.error(f"Kunde inte ladda standard kokbok: {str(e)}")

# Sidebar för uppladdningar
with st.sidebar:
    
    # Visa aktuell kokbok
    if st.session_state.pdf_processed:
        st.success(f"**{st.session_state.current_pdf_name}**")
        st.caption(f"({len(st.session_state.chunks)} stycken)")
    else:
        st.info("Ingen kokbok laddad")
    
    # Möjlighet att byta kokbok
    with st.expander("Byt kokbok"):
        uploaded_file = st.file_uploader(
            "Välj en annan PDF-kokbok",
            type=['pdf'],
            help="Ladda upp en annan PDF-kokbok för att byta."
        )
        
        if uploaded_file is not None:
            if st.button("Använd denna kokbok", use_container_width=True):
                with st.spinner("Bearbetar ny kokbok..."):
                    try:
                        text, chunks, embeddings = process_pdf(uploaded_file)
                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.pdf_processed = True
                        st.session_state.document_length = len(text)
                        st.session_state.current_pdf_name = uploaded_file.name
                        st.success("Kokbok bytt!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fel vid bearbetning: {str(e)}")
        
        # Knapp för att återställa till standard
        if st.session_state.current_pdf_name != "USU Student Cookbook":
            if st.button("Återställ till USU Cookbook", use_container_width=True):
                with st.spinner("Laddar standard kokbok..."):
                    try:
                        text, chunks, embeddings = process_pdf_from_path(DEFAULT_PDF_PATH)
                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.pdf_processed = True
                        st.session_state.document_length = len(text)
                        st.session_state.current_pdf_name = "USU Student Cookbook"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fel: {str(e)}")
    
    st.divider()
    
    st.header("Ladda upp Ingredienser")
    
    # Val mellan kamera och filuppladdning
    input_method = st.radio(
        "Välj metod:",
        ["Fota kylen", "Ladda upp bild"],
        horizontal=True
    )
    
    if input_method == "Fota kylen":
        camera_image = st.camera_input("Fota dina ingredienser")
        if camera_image is not None:
            st.session_state.uploaded_image = Image.open(camera_image)
    else:
        uploaded_image = st.file_uploader(
            "Välj en bild",
            type=['jpg', 'png', 'jpeg'],
            help="Ladda upp en bild på dina ingredienser."
        )
        if uploaded_image is not None:
            st.session_state.uploaded_image = Image.open(uploaded_image)
    
    # Visa bild-status och förhandsvisning
    if st.session_state.uploaded_image is not None:
        st.success("Bild laddad")
        st.image(st.session_state.uploaded_image, caption="Din bild", use_container_width=True)
    else:
        st.info("Ingen bild laddad")
    
    st.divider()
    
    if st.button("Börja om (behåll kokbok)", use_container_width=True):
        # Behåll kokboken, rensa bara bild och recept
        st.session_state.uploaded_image = None
        st.session_state.identified_ingredients = None
        st.session_state.generated_recipe = None
        st.rerun()

# Huvudområde
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Status")
    
    # Visa status-indikatorer
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.pdf_processed:
            st.success("Kokbok redo ✅")
        else:
            st.warning("Ladda upp kokbok")
    
    with status_col2:
        if st.session_state.uploaded_image is not None:
            st.success("Bild redo ✅")
        else:
            st.warning("Ladda upp bild")

with col2:
    st.subheader("Skapa Recept")
    
    # Kontrollera om allt är redo
    ready_to_cook = st.session_state.pdf_processed and st.session_state.uploaded_image is not None
    
    if not ready_to_cook:
        st.info("Ladda upp både en kokbok och en bild för att skapa recept.")
    else:
        if st.button("Skapa Recept", use_container_width=True, type="primary"):
            try:
                # Steg 1: Identifiera ingredienser
                with st.spinner("Tittar på bilden..."):
                    ingredients = identify_ingredients(st.session_state.uploaded_image)
                    st.session_state.identified_ingredients = ingredients
                
                # Vänta 1 sekund för att undvika rate limiting (429-fel)
                time.sleep(1)
                
                # Steg 2: Sök i kokboken
                with st.spinner("Söker i kokboken..."):
                    relevant_chunks = semantic_search(
                        ingredients,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        k=5
                    )
                
                # Vänta 1 sekund för att undvika rate limiting (429-fel)
                time.sleep(1)
                
                # Steg 3: Generera recept
                with st.spinner("Skriver recept..."):
                    recipe = generate_recipe(ingredients, relevant_chunks)
                    st.session_state.generated_recipe = recipe
                
            except Exception as e:
                st.error(f"Fel vid receptgenerering: {str(e)}")

# Visa resultat
st.divider()

if st.session_state.identified_ingredients:
    st.subheader("Hittade Ingredienser")
    st.info(st.session_state.identified_ingredients)

if st.session_state.generated_recipe:
    st.subheader("Ditt Recept")
    st.markdown(st.session_state.generated_recipe)

