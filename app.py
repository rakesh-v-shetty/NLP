import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# Removed google.generativeai as we are no longer using the API
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # Import transformers components
import torch # For device management

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# No GEMINI_API_KEY needed as we are using a local model
# GEMINI_API_KEY = 'YOUR_ACTUAL_GEMINI_API_KEY_HERE'
# genai.configure(api_key=GEMINI_API_KEY) # This line is removed

CHUNKS_FILE_PATH = 'medical_text_chunks.pkl'
FAISS_INDEX_FILE_PATH = 'medical_faiss_index.bin'
LOCAL_LLM_PATH = 'local_llm_tinyllama' # Path to the downloaded LLM folder

text_chunks_db = []
faiss_index = None
embedding_model = None
local_llm_tokenizer = None
local_llm_model = None
text_generation_pipeline = None # Hugging Face pipeline for easier generation

# --- Load Pre-processed Data and Local LLM ---
def load_preprocessed_data_and_llm():
    global text_chunks_db, faiss_index, embedding_model, local_llm_tokenizer, local_llm_model, text_generation_pipeline

    print("Loading Sentence-Transformer embedding model for backend...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading Sentence-Transformer model: {e}")
        print("Please ensure 'sentence-transformers' library is installed (`pip install sentence-transformers`).")
        return False

    print(f"Loading text chunks from {CHUNKS_FILE_PATH}...")
    try:
        with open(CHUNKS_FILE_PATH, 'rb') as f:
            text_chunks_db = pickle.load(f)
        print(f"Loaded {len(text_chunks_db)} text chunks.")
    except FileNotFoundError:
        print(f"Error: {CHUNKS_FILE_PATH} not found. Please ensure it's in the correct directory.")
        text_chunks_db = []
        return False
    except Exception as e:
        print(f"Error loading text chunks: {e}")
        text_chunks_db = []
        return False

    print(f"Loading FAISS index from {FAISS_INDEX_FILE_PATH}...")
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_FILE_PATH)
        print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
    except FileNotFoundError:
        print(f"Error: {FAISS_INDEX_FILE_PATH} not found. Please ensure it's in the correct directory.")
        faiss_index = None
        return False
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        faiss_index = None
        return False

    print(f"\n--- Loading Local Generative LLM from {LOCAL_LLM_PATH} ---")
    try:
        # Check for CUDA device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for LLM: {device}")

        local_llm_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH)
        local_llm_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True # Helps with memory on CPU
        ).to(device) # Move model to appropriate device

        # Create a text generation pipeline for easier inference
        text_generation_pipeline = pipeline(
            "text-generation",
            model=local_llm_model,
            tokenizer=local_llm_tokenizer,
            device=0 if device == "cuda" else -1 # 0 for first GPU, -1 for CPU
        )
        print("Local LLM and tokenizer loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Local LLM model not found at {LOCAL_LLM_PATH}.")
        print("Please ensure the 'local_llm_tinyllama' folder is in the correct directory.")
        local_llm_tokenizer = None
        local_llm_model = None
        text_generation_pipeline = None
        return False
    except Exception as e:
        print(f"Error loading local LLM: {e}")
        local_llm_tokenizer = None
        local_llm_model = None
        text_generation_pipeline = None
        return False

    return True

# Load data and LLM when the Flask app starts
with app.app_context():
    if not load_preprocessed_data_and_llm():
        print("CRITICAL: Failed to load all necessary components. The application may not function correctly.")

# --- RAG (Retrieval-Augmented Generation) Logic ---
def retrieve_relevant_context(query, top_k=5):
    if faiss_index is None or embedding_model is None or not text_chunks_db:
        print("RAG components not fully loaded. Cannot retrieve context.")
        return []

    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        D, I = faiss_index.search(query_embedding, top_k)

        relevant_chunks = []
        for i in I[0]:
            if i != -1:
                relevant_chunks.append(text_chunks_db[i]['text'])
        return relevant_chunks
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return []

# --- Flask Route to Serve Frontend HTML ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Flask API Endpoint for Asking Questions ---
@app.route('/ask', methods=['POST'])
def ask_medpromptai():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if text_generation_pipeline is None:
        return jsonify({"error": "Local LLM is not loaded. Cannot generate response."}), 500

    try:
        relevant_contexts = retrieve_relevant_context(question, top_k=5)

        context_string = ""
        if relevant_contexts:
            context_string = "\nRelevant Medical Information (from Textbooks):\n" + "\n\n".join(relevant_contexts) + "\n\n"
        else:
            context_string = "No highly relevant context found in the provided medical textbooks. Relying on general knowledge from the AI model.\n\n"

        # --- PROMPT FOR LOCAL LLM ---
        # The prompt is designed to guide the small LLM to act as a chatbot
        # and use the provided context.
        prompt = f"""
        You are a helpful, knowledgeable, and safe AI medical chatbot. Your primary goal is to provide accurate general health information based *only* on the provided contextual information from trusted medical textbooks.
        Respond in complete, grammatically correct sentences. Maintain a friendly and conversational tone.
        Do not use outside knowledge unless explicitly stated that the provided context is insufficient.
        Crucially, never provide a direct medical diagnosis, prescribe treatment, or give personalized medical advice.
        Always state if the provided context is insufficient to fully answer the question.
        At the end of your response, always remind the user to consult a qualified healthcare professional for personalized medical advice.

        {context_string}

        Based *only* on the 'Relevant Medical Information' provided above (if any), please answer the following medical question in a conversational manner:
        Question: {question}
        """

        # Generate response using the local LLM pipeline
        # max_new_tokens: controls the length of the generated response
        # num_return_sequences: how many different responses to generate (we take the first)
        # temperature: creativity (lower for more factual, higher for more creative)
        # do_sample: set to True to enable sampling (using temperature)
        # top_k, top_p: sampling strategies
        outputs = text_generation_pipeline(
            prompt,
            max_new_tokens=500, # Adjust as needed for desired response length
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=local_llm_tokenizer.eos_token_id # Important for handling padding
        )
        ai_response_text = outputs[0]['generated_text']

        # The model might repeat the prompt, so we need to clean it
        if ai_response_text.startswith(prompt):
            ai_response_text = ai_response_text[len(prompt):].strip()

        # Basic Hallucination/Safety Filtering (Post-processing)
        safety_keywords = [
            "your diagnosis is", "you have", "I diagnose", "prescribe", "take this medication",
            "treatment for you is", "your condition is", "I recommend you take", "cure for",
            "you should take", "my diagnosis is"
        ]
        flagged_for_review = False
        for keyword in safety_keywords:
            if keyword in ai_response_text.lower():
                flagged_for_review = True
                break

        disclaimer_text = "\n\n--- DISCLAIMER ---\nThis AI tool provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical advice."
        if disclaimer_text.lower().strip() not in ai_response_text.lower().strip():
            ai_response_text += disclaimer_text

        return jsonify({
            "answer": ai_response_text,
            "retrieved_context": relevant_contexts,
            "flagged_for_review": flagged_for_review
        })

    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again or re-check the backend server."}), 500

# --- Running the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)