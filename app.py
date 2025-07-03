import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import torch

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# You MUST replace 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' with your actual Gemini API key.
# It is highly recommended to load this from an environment variable for security.
GEMINI_API_KEY = 'AIzaSyDSQc5d5zPd2IXP5kyGF5MNfcACV8JAJHc' #

if GEMINI_API_KEY == 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' or not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set or is still the placeholder. "
          "Please replace 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' with your actual API key "
          "or set it as an environment variable (e.g., GEMINI_API_KEY='your_key').")
    # For a real deployment, you might want to exit or raise an error here.
    # For local testing, we'll allow it to proceed with a warning.

genai.configure(api_key=GEMINI_API_KEY) #

CHUNKS_FILE_PATH = os.path.join('medprompt_ai_data', 'medical_text_chunks.pkl') #
FAISS_INDEX_FILE_PATH = os.path.join('medprompt_ai_data', 'medical_faiss_index.bin') #

text_chunks_db = []
faiss_index = None
embedding_model = None
gemini_model = None

# --- Load Pre-processed Data and Configure Gemini LLM ---
def load_preprocessed_data_and_configure_gemini():
    global text_chunks_db, faiss_index, embedding_model, gemini_model

    print("Loading pre-processed data...") #
    try:
        if not os.path.exists(CHUNKS_FILE_PATH): #
            raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE_PATH}. Run model.py first.") #
        with open(CHUNKS_FILE_PATH, 'rb') as f: #
            text_chunks_db = pickle.load(f) #
        print(f"Loaded {len(text_chunks_db)} text chunks.") #

        if not os.path.exists(FAISS_INDEX_FILE_PATH): #
            raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_FILE_PATH}. Run model.py first.") #
        faiss_index = faiss.read_index(FAISS_INDEX_FILE_PATH) #
        print(f"Loaded FAISS index with {faiss_index.ntotal} vectors.") #

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2') #
        print("Loaded Embedding Model: all-MiniLM-L6-v2.") #

    except FileNotFoundError as e: #
        print(f"Error loading pre-processed data: {e}. Make sure you run model.py first to generate these files.") #
        exit(1)
    except Exception as e: #
        print(f"An unexpected error occurred during data loading: {e}") #
        exit(1)

    print("\nConfiguring Gemini LLM...") #
    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') #
        print("Gemini 'gemini-2.0-flash' model configured successfully.") #
    except Exception as e: #
        print(f"Error configuring Gemini LLM: {e}. Please check your GEMINI_API_KEY and network connection.") #
        exit(1)

# Call the loading function when the app starts
load_preprocessed_data_and_configure_gemini() #

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html') #

@app.route('/ask', methods=['POST'])
def ask_medprompt_ai():
    user_question = request.json.get('question') #
    if not user_question: #
        return jsonify({"error": "No question provided"}), 400 #

    try:
        if not embedding_model or not faiss_index or not gemini_model: #
            return jsonify({"error": "MedPromptAI is still initializing or encountered a critical loading error. Please wait a moment and try again. If the issue persists, check server logs."}), 503 #

        question_embedding = embedding_model.encode([user_question]).astype('float32') #

        D, I = faiss_index.search(question_embedding, k=5) # k=5 for top 5 most relevant chunks
        relevant_indices = I[0] #

        relevant_contexts = [text_chunks_db[idx]['text'] for idx in relevant_indices if idx < len(text_chunks_db)] #
        context_str = "\n".join(relevant_contexts) #

        prompt = (
            "You are a medical assistant chatbot. "
            "Based on the following medical context, answer the user's question concisely and accurately. "
            "If the information is not directly available in the provided context, state that you cannot answer based on the given information. "
            "Do not make up information. Do not mention the context directly in your answer, just use the information.\n\n"
            "Medical Context:\n" + context_str + "\n\n"
            "User Question: " + user_question + "\n\n"
            "Answer:"
        ) #
        
        response = gemini_model.generate_content( #
            prompt,
            generation_config=genai.types.GenerationConfig( #
                temperature=0.7, #
                max_output_tokens=512, #
            ),
            safety_settings=[ #
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, #
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}, #
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}, #
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}, #
            ]
        )
        
        ai_response_text = "" #
        try:
            ai_response_text = response.text.strip() #
        except ValueError as e: #
            if 'safety_ratings' in response.candidates[0]: #
                safety_feedback = "The response was blocked due to safety concerns." #
                for rating in response.candidates[0].safety_ratings: #
                    safety_feedback += f"\n- {rating.category}: {rating.probability}" #
                ai_response_text = f"I apologize, I cannot generate a response for that query. {safety_feedback}" #
            else:
                ai_response_text = "I apologize, I could not generate a response for your query." #
            print(f"Gemini API generation error or block: {e}") #

        safety_keywords = [
            "your diagnosis is", "you have", "I diagnose", "prescribe", "take this medication",
            "treatment for you is", "I recommend you take", "cure for",
            "you should take", "my diagnosis is"
        ] #
        flagged_for_review = False #
        for keyword in safety_keywords: #
            if keyword in ai_response_text.lower(): #
                flagged_for_review = True #
                break #

        # REMOVED: Disclaimer text appending logic
        # disclaimer_text = "\n\n--- DISCLAIMER ---\nThis AI tool provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical advice."
        # if disclaimer_text.lower().strip().replace(' ', '') not in ai_response_text.lower().strip().replace(' ', ''):
        #     ai_response_text += disclaimer_text

        return jsonify({
            "answer": ai_response_text,
            "flagged_for_review": flagged_for_review
        })

    except Exception as e:
        print(f"Error in /ask endpoint: {e}") #
        return jsonify({"error": "An error occurred while processing your request. Please try again or re-check the backend server."}), 500 #

# --- Running the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) #
