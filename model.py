# @title MedPromptAI: Colab Pre-processing & Local LLM Preparation (NLTK punkt_tab FIX)
#
# This script is designed to be run in a Google Colab environment.
# It processes medical textbook files, generates semantic embeddings,
# builds a FAISS index, and downloads a small generative LLM for local use.

# --- 1. Install Necessary Libraries ---
!pip install sentence-transformers faiss-cpu transformers accelerate bitsandbytes torch

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.notebook import tqdm
import torch # For checking CUDA availability and device management

# --- 2. Mount Google Drive (Optional but Recommended) ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/medprompt_ai_data'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Google Drive mounted. Data will be saved to: {SAVE_DIR}")
except ImportError:
    print("Not in Google Colab environment or Drive not mounted. Saving to local Colab session storage.")
    SAVE_DIR = './medprompt_ai_data'
    os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. Define Textbook File Paths ---
TEXTBOOK_FILENAMES = [
    "Anatomy_Gray.txt", "Biochemistry_Lippincott.txt", "Cell_Biology_Alberts.txt",
    "First_Aid_Step1.txt", "First_Aid_Step2.txt", "Gynecology_Novak.txt",
    "Histology_Ross.txt", "Immunology_Janeway.txt", "InternalMed_Harrison.txt",
    "Neurology_Adams.txt", "Obstentrics_Williams.txt", "Pathology_Robbins.txt",
    "Pathoma_Husain.txt", "Pediatrics_Nelson.txt", "Pharmacology_Katzung.txt",
    "Physiology_Levy.txt", "Psichiatry_DSM-5.txt", "Surgery_Schwartz.txt"
]

# --- 4. Initialize Embedding Model ---
print("Loading Sentence-Transformer embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# --- NLTK Download and Setup (Moved to top for robustness) ---
use_nltk_sent_tokenize = False
try:
    import nltk
    print("Attempting to download NLTK 'punkt' tokenizer data...")
    # Try downloading 'punkt_tab' specifically as per the error message
    try:
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        # Test if it works with a simple sentence
        _ = sent_tokenize("This is a test sentence.")
        print("NLTK 'punkt_tab' tokenizer loaded successfully.")
        use_nltk_sent_tokenize = True
    except LookupError:
        print("NLTK 'punkt_tab' data not found. Attempting general 'punkt' download.")
        nltk.download('punkt', quiet=True)
        try:
            from nltk.tokenize import sent_tokenize
            _ = sent_tokenize("This is another test sentence.")
            print("NLTK 'punkt' tokenizer loaded successfully after general download.")
            use_nltk_sent_tokenize = True
        except LookupError:
            print("NLTK 'punkt' data still not found. Falling back to basic sentence splitting.")
except ImportError:
    print("NLTK library not installed. Falling back to basic sentence splitting.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK setup: {e}. Falling back to basic sentence splitting.")
# --- End NLTK Setup ---


# --- 5. Text Loading, Chunking, and Embedding Generation ---
text_chunks_with_embeddings = []
documents_content = {}

async def fetch_file_content(filename):
    file_path = os.path.join("/content/", filename) # Adjust if files are in a subfolder
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure '{filename}' is uploaded to Colab or path is correct.")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

print("Fetching content of medical textbooks...")
for filename in tqdm(TEXTBOOK_FILENAMES, desc="Fetching files"):
    content = await fetch_file_content(filename)
    if content:
        documents_content[filename] = content

print("Processing and embedding text chunks...")
for filename, text in tqdm(documents_content.items(), desc="Processing textbooks"):
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if para.strip():
            # Use NLTK sent_tokenize if available and paragraph is long, otherwise fallback
            if len(para) > 500 and use_nltk_sent_tokenize:
                sentences = sent_tokenize(para)
                # Filter out empty strings that might result from tokenization
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                # Fallback to simple split by common sentence terminators
                sentences = []
                temp_split = para.split('.')
                for s_part in temp_split:
                    s_part_q = s_part.split('?')
                    for sq_part in s_part_q:
                        s_part_e = sq_part.split('!')
                        for se_part in s_part_e:
                            trimmed = se_part.strip()
                            if trimmed:
                                sentences.append(trimmed)

            for sentence in sentences:
                trimmed_sentence = sentence.strip()
                if trimmed_sentence:
                    text_chunks_with_embeddings.append({
                        'text': trimmed_sentence,
                        'source_file': filename
                    })

print(f"Generating embeddings for {len(text_chunks_with_embeddings)} chunks...")
chunk_texts = [chunk['text'] for chunk in text_chunks_with_embeddings]
embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

for i, embedding in enumerate(embeddings):
    text_chunks_with_embeddings[i]['embedding'] = embedding

print("Embeddings generated.")

# --- 6. Build FAISS Index ---
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")

# --- 7. Save Processed Data and Index ---
chunks_file_path = os.path.join(SAVE_DIR, 'medical_text_chunks.pkl')
with open(chunks_file_path, 'wb') as f:
    pickle.dump(text_chunks_with_embeddings, f)
print(f"Text chunks with embeddings saved to: {chunks_file_path}")

faiss_index_file_path = os.path.join(SAVE_DIR, 'medical_faiss_index.bin')
faiss.write_index(index, faiss_index_file_path)
print(f"FAISS index saved to: {faiss_index_file_path}")

# --- 8. Download and Save Local Generative LLM ---
print("\n--- Preparing Local Generative LLM (TinyLlama) ---")
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_SAVE_PATH = os.path.join(SAVE_DIR, "local_llm_tinyllama")

print(f"Downloading model: {LLM_MODEL_NAME} and saving to {LLM_SAVE_PATH}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
tokenizer.save_pretrained(LLM_SAVE_PATH)
print("Tokenizer saved.")

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
model.save_pretrained(LLM_SAVE_PATH)
print("Model saved.")

print(f"\nLocal LLM '{LLM_MODEL_NAME}' downloaded and saved to: {LLM_SAVE_PATH}")
print("\nPre-processing and LLM preparation complete! You can now download the following:")
print(f"- {chunks_file_path}")
print(f"- {faiss_index_file_path}")
print(f"- The entire folder: {LLM_SAVE_PATH} (contains model weights and tokenizer files)")
print("\nRemember to place these files/folder in the same directory as your Flask backend script.")