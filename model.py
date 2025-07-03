# @title MedPromptAI: Colab Pre-processing & Local LLM Preparation
#
# This script is designed to be run in a Google Colab environment.
# It processes medical textbook files, generates semantic embeddings,
# builds a FAISS index, and downloads a small generative LLM for local use.

# --- 1. Install Necessary Libraries ---
# Ensure you have all necessary libraries, especially for BitsAndBytes if using CUDA
# !pip install sentence-transformers faiss-cpu transformers accelerate bitsandbytes torch nltk huggingface_hub

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.notebook import tqdm
import torch # For checking CUDA availability and device management
import re # For regex-based sentence splitting fallback
import sys # For sys.exit()

# --- Hugging Face Login (NEW ADDITION) ---
from huggingface_hub import login

print("--- Hugging Face Authentication ---")
print("The 'mistralai/Mistral-7B-Instruct-v0.2' model is gated.")
print("You need to: ")
print("1. Have a Hugging Face account.")
print("2. Accept the model's terms on its Hugging Face page: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
print("3. Generate a Hugging Face access token (with 'read' access) from your settings: https://huggingface.co/settings/tokens")
print("4. Log in using the token below:")

try:
    # This will prompt you to enter your Hugging Face token in the Colab output
    login()
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to log in to Hugging Face: {e}")
    print("Model download may fail if not authenticated.")
print("-----------------------------------\n")

# --- NLTK Setup (Revised based on user suggestion) ---
use_nltk_sent_tokenize = False
try:
    import nltk
    print("Attempting to download NLTK 'punkt_tab' tokenizer data...")
    try:
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        _ = sent_tokenize("This is a test sentence for punkt_tab.") # Test if it works
        print("NLTK 'punkt_tab' tokenizer loaded successfully.")
        use_nltk_sent_tokenize = True
    except LookupError:
        print("NLTK 'punkt_tab' data not found. Attempting general 'punkt' download.")
        nltk.download('punkt', quiet=True)
        try:
            from nltk.tokenize import sent_tokenize
            _ = sent_tokenize("This is another test sentence for punkt.") # Test if it works
            print("NLTK 'punkt' tokenizer loaded successfully after general download.")
            use_nltk_sent_tokenize = True
        except LookupError:
            print("NLTK 'punkt' data still not found. Falling back to basic sentence splitting.")
except ImportError:
    print("NLTK library not installed. Falling back to basic sentence splitting.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK setup: {e}. Falling back to basic sentence splitting.")
# --- End NLTK Setup ---


# --- 2. Mount Google Drive (Optional but Recommended) ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/medprompt_ai_data'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Google Drive mounted. Data will be saved to: {SAVE_DIR}")
except (ImportError, Exception) as e: # Catch broader exceptions for mount issues
    print(f"Not in Google Colab environment or Drive not mounted: {e}. Saving to local session storage.")
    SAVE_DIR = './medprompt_ai_data'
    os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. Define Textbook File Paths ---
# Ensure these files are placed in a 'Medical_Textbooks' folder
# relative to where you run this script, or adjust the filepath.
TEXTBOOK_FILENAMES = [
    "Anatomy_Gray.txt", "Biochemistry_Lippincott.txt", "Cell_Biology_Alberts.txt",
    "First_Aid_Step1.txt", "First_Aid_Step2.txt", "Gynecology_Novak.txt",
    "Histology_Ross.txt", "Immunology_Janeway.txt", "InternalMed_Harrison.txt",
    "Neurology_Adams.txt", "Obstentrics_Williams.txt", "Pathology_Robbins.txt",
    "Pathoma_Husain.txt", "Pediatrics_Nelson.txt", "Pharmacology_Katzung.txt",
    "Physiology_Levy.txt", "Psichiatry_DSM-5.txt", "Surgery_Schwartz.txt"
]

# --- 4. Process Textbooks, Generate Chunks, and Embeddings ---
text_chunks = []
current_file_sentences_buffer = [] # Buffer to hold sentences for overlap processing

print("\n--- Processing Textbooks and Generating Chunks ---")
for filename in tqdm(TEXTBOOK_FILENAMES, desc="Processing textbooks"):
    filepath = os.path.join('Medical_Textbooks', filename) # Assuming textbooks are in this folder
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}. Skipping.")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    current_file_sentences_buffer = [] # Reset for each new file
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip(): # Ensure paragraph is not empty
            if use_nltk_sent_tokenize:
                sentences = [s.strip() for s in sent_tokenize(para) if s.strip()]
            else:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                # print("Using simple regex-based sentence splitting (NLTK not available).") # Optional: uncomment for debugging

            current_file_sentences_buffer.extend(sentences) # Add all sentences from this paragraph/file

    # Now, process sentences with overlap for this file
    # Define how many previous sentences to include (e.g., 1 for "current + 1 previous")
    OVERLAP_SENTENCES = 1 # Adjust as needed (1 or 2 is common for medical text)

    for i, sentence in enumerate(current_file_sentences_buffer):
        # Optional: Filter out very short, non-informative sentences before creating chunks
        if len(sentence.strip()) < 20: # Minimum character length for a chunk
            continue

        chunk_content = sentence
        # Add previous sentences for context
        for j in range(1, OVERLAP_SENTENCES + 1):
            if i - j >= 0:
                chunk_content = current_file_sentences_buffer[i - j] + " " + chunk_content

        text_chunks.append({"text": chunk_content, "source": filename})

print(f"\nTotal text chunks created: {len(text_chunks)}")

# --- 5. Generate Embeddings for Chunks ---
print("\n--- Generating Embeddings ---")
# Using 'all-MiniLM-L6-v2' for efficient and effective semantic embeddings.
# This model provides a good balance for general-purpose similarity search.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedding_model.encode([chunk['text'] for chunk in text_chunks], show_progress_bar=True)
print(f"Generated {len(chunk_embeddings)} embeddings of dimension {chunk_embeddings.shape[1]}.")

# --- 6. Build FAISS Index ---
print("\n--- Building FAISS Index ---")
embedding_dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension) # L2 distance for similarity
index.add(np.array(chunk_embeddings).astype('float32'))
print(f"FAISS index built with {index.ntotal} vectors.")

# --- 7. Save Processed Data ---
print("\n--- Saving Processed Data ---")
chunks_file_path = os.path.join(SAVE_DIR, 'medical_text_chunks.pkl')
with open(chunks_file_path, 'wb') as f:
    pickle.dump(text_chunks, f)
print(f"Text chunks saved to: {chunks_file_path}")

faiss_index_file_path = os.path.join(SAVE_DIR, 'medical_faiss_index.bin')
faiss.write_index(index, faiss_index_file_path)
print(f"FAISS index saved to: {faiss_index_file_path}")

