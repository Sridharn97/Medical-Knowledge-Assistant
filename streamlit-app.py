import streamlit as st
import ollama
import os

# Set page configuration
st.set_page_config(page_title="Medical Knowledge Assistant", layout="centered")
st.title("Medical Knowledge Assistant")
st.write("Ask questions based on your local medical knowledge dataset. Not a substitute for professional medical advice.")

# Model Configurations
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:1b'

def ensure_model(model_name: str) -> None:
    try:
        local_models = [m.get('name') for m in ollama.list().get('models', [])]
        if model_name in local_models:
            return
    except Exception:
        # If `ollama list` fails for any reason, fall back to pulling.
        pass

    with st.spinner(f'Pulling Ollama model: {model_name}'):
        try:
            ollama.pull(model_name)
        except Exception as e:
            st.error(
                f'Failed to pull model "{model_name}". '
                f'Try running `ollama pull {model_name}` in a terminal. '
                f'Original error: {e}'
            )
            raise


ensure_model(EMBEDDING_MODEL)
ensure_model(LANGUAGE_MODEL)

# --- 1. Vector DB Setup & Caching ---
@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    dataset_path = 'medical.txt'
    
    # Check if file exists to prevent crashing
    if not os.path.exists(dataset_path):
        # Creating a fallback file for demonstration if it doesn't exist
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.write("Disclaimer: This dataset is for informational purposes only and is not a substitute for professional medical advice.\n")
            f.write("High blood pressure (hypertension) increases risk of heart disease and stroke.\n")
            f.write("Fever is a common sign of infection but may also arise from non-infectious causes.\n")
            f.write("For life-threatening emergencies (severe chest pain, uncontrolled bleeding, sudden unconsciousness), seek immediate medical care.\n")
            f.write("Maintain a balanced diet, regular exercise, and adequate sleep for overall health.\n")

    with open(dataset_path, 'r', encoding='utf-8', errors='replace') as file:
        dataset = file.readlines()
    
    vector_db = []
    
    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if not chunk:
            continue
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return []
        progress_bar.progress((i + 1) / len(dataset))
        
    # Clear the loading indicators when done
    status_text.empty()
    progress_bar.empty()
    return vector_db

# Initialize the database
with st.spinner("Initializing Vector Database and generating embeddings..."):
    VECTOR_DB = initialize_vector_db()

# --- 2. Helper Functions ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if int(norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    except Exception as e:
        st.error(f"Failed to generate query embedding: {e}")
        return []
        
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- 3. UI and Interaction ---

# Sidebar to show the retrieved context chunks behind the scenes
with st.sidebar:
    st.header("Database Info")
    st.success(f"Loaded {len(VECTOR_DB)} items from dataset.")
    st.markdown("---")
    st.subheader("Retrieved Context Window")
    context_placeholder = st.empty()
    context_placeholder.info("Ask a question to see the matching context chunks here!")

# User Input
input_query = st.text_input("Ask me a question about medical topics:", placeholder="e.g., What are common symptoms of hypertension?")

if input_query:
    # Perform Retrieval
    retrieved_knowledge = retrieve(input_query)
    
    # Update the sidebar dynamically to display retrieved chunks and their scores
    with context_placeholder.container():
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f"**Score:** `{similarity:.2f}`\n\n*{chunk}*")
            st.markdown("---")

    # Construct the instruction prompt
    formatted_context = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
    instruction_prompt = f'''
You are a helpful chatbot. Use only the following pieces of context to answer the question.
Don't make up any new information:
{formatted_context}
'''

    st.subheader("Response:")
    
    # Dynamic placeholder for streaming the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )
        
        # Iterate over stream chunks and dynamically update the UI
        for chunk in stream:
            token = chunk['message']['content']
            full_response += token
            response_placeholder.markdown(full_response + "▌")
            
        # Final update to remove the cursor icon
        response_placeholder.markdown(full_response)
        
    except Exception as e:
        st.error(f"Error generating chat response: {e}")
