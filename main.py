# import os
# import uuid
# import fitz  # PyMuPDF: for extracting text from PDF documents
# import docx2txt  # For extracting text from DOCX documents
# import requests  # To download files from URLs
# import tempfile  # For creating temporary local files
# import faiss  # Facebook AI Similarity Search - used for vector indexing
# import numpy as np  # For numerical operations
# import google.generativeai as genai  # Google Gemini LLM API
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer  # Sentence Embedding model
# from typing import List
# from dotenv import load_dotenv  # For loading environment variables

# # -------- STEP 0: LOAD ENVIRONMENT VARIABLES -------- #
# load_dotenv()  # Load .env file from project root

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

# if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
#     raise RuntimeError("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN in environment variables.")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Load sentence embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
# dimension = 384

# # FastAPI app instance
# app = FastAPI(title="LLM Query Retrieval")

# # HTTP Bearer security scheme
# security = HTTPBearer()

# # Initialize global FAISS index
# index = faiss.IndexFlatL2(dimension)
# id_to_chunk = []

# # -------- AUTH CHECK -------- #
# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.credentials != TEAM_AUTH_TOKEN:
#         raise HTTPException(status_code=403, detail="Unauthorized token.")

# # -------- INPUT / OUTPUT MODELS -------- #
# class QueryInput(BaseModel):
#     documents: str
#     questions: List[str]

# class QueryOutput(BaseModel):
#     answers: List[str]

# # -------- TEXT EXTRACTION FROM URL -------- #
# def extract_text_from_url(url: str) -> str:
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to fetch document.")

#     ext = url.split('.')[-1].split('?')[0]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
#         tmp_file.write(response.content)
#         path = tmp_file.name

#     if ext == "pdf":
#         doc = fitz.open(path)
#         return " ".join(page.get_text() for page in doc)
#     elif ext == "docx":
#         return docx2txt.process(path)
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported or unknown file type.")

# # -------- CHUNK DOCUMENT INTO TEXT BLOCKS -------- #
# def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = words[i:i + chunk_size]
#         chunks.append(" ".join(chunk))
#     return chunks

# # -------- BUILD FAISS SEMANTIC INDEX -------- #
# def build_faiss_index(chunks: List[str]):
#     global index, id_to_chunk
#     index.reset()
#     id_to_chunk.clear()
#     embeddings = model.encode(chunks)
#     index.add(np.array(embeddings))
#     id_to_chunk.extend(chunks)

# # -------- RETRIEVE TOP-K CHUNKS -------- #
# def retrieve_relevant_chunks(query: str, top_k: int = 10) -> List[str]:
#     query_vec = model.encode([query])
#     D, I = index.search(np.array(query_vec), top_k)
#     return [id_to_chunk[i] for i in I[0]]

# # -------- GEMINI ANSWER GENERATOR -------- #
# def answer_with_gemini(query: str, context_chunks: List[str]) -> str:
#     context = "\n\n".join(context_chunks)
#     prompt = f"""
# You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

# Instructions:
# 1. Answer in complete sentences that **mirror the style and content of the original policy language**.
# 2. If a specific duration, condition, or percentage is mentioned, always include it.
# 3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
# 4. If no exact match exists, extract the most relevant and complete information available.
# 5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
# 6. Do not say "based on the context above" or "according to the document".
# 7. Be concise but **do not omit key legal terms or numbers**.
# 8. Do not generate an answer unless the information is supported in the context.

# Now answer the following question:

# Question: {query}

# Context:
# {context}
# """
#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

# # -------- MAIN API ROUTE WITH AUTH -------- #
# @app.post("/api/v1/hackrx/run", response_model=QueryOutput)
# async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     try:
#         raw_text = extract_text_from_url(input_data.documents)
#         chunks = chunk_text(raw_text)
#         build_faiss_index(chunks)

#         final_answers = []
#         for q in input_data.questions:
#             top_chunks = retrieve_relevant_chunks(q)
#             answer = answer_with_gemini(q, top_chunks)
#             final_answers.append(answer)

#         return {"answers": final_answers}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
   
import os
import tempfile
import requests
import email.message
import logging # Re-import logging explicitly
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as extract_pdf_text # Note: This uses pdfminer.six
import docx2txt
import faiss
import numpy as np
import google.generativeai as genai
from bs4 import BeautifulSoup # For parsing HTML content in emails
# import uvicorn # Not typically imported at the top for Gunicorn/Uvicorn on Render

# Configure logging at the top
logging.basicConfig(level=logging.INFO) # Set to INFO for production
logger = logging.getLogger("uvicorn.error") # Use the standard uvicorn logger

# Initialize environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
    logger.error("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN. Please set them in your .env file.")
    raise RuntimeError("Missing credentials. Please check environment variables.")

# Configure Gemini and load models GLOBALLY (only once at app startup)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash') # LOAD ONCE

# Load sentence embedding model GLOBALLY (only once at app startup)
# "paraphrase-albert-small-v2" is indeed a smaller model (approx 42MB)
embedding_model = SentenceTransformer("paraphrase-albert-small-v2") # LOAD ONCE
dimension = embedding_model.get_sentence_embedding_dimension() # Get dimension dynamically

# FastAPI app instance
app = FastAPI(title="LLM Query Retrieval for Hackathon")

# HTTP Bearer security scheme
security = HTTPBearer()

# Initialize global FAISS index (reset for each request, but declared globally)
index = faiss.IndexFlatL2(dimension)
id_to_chunk_store = [] # Renamed to avoid conflict with `model` function parameter

# ========================== SCHEMAS ==========================
class QueryInput(BaseModel):
    documents: str
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[str]

# ========================== FILE EXTRACTION ==========================
def extract_text_from_file(path: str, ext: str) -> str:
    """Extracts text from PDF or DOCX files."""
    try:
        if ext == "pdf":
            return extract_pdf_text(path)
        elif ext == "docx":
            return docx2txt.process(path)
        else:
            raise ValueError(f"Unsupported file extension for direct extraction: {ext}")
    except Exception as e:
        logger.error(f"Error extracting text from {ext} file '{path}': {e}")
        raise

def extract_text_from_eml(path: str) -> str:
    """Extracts text from email body (plain, HTML) and attachments (PDF, DOCX)."""
    extracted_parts = []
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f)

        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition() # Can be None for body parts
            filename = part.get_filename()
            payload = part.get_payload(decode=True)

            # Prioritize attachments first
            if disposition == "attachment" and filename:
                ext = filename.split(".")[-1].lower()
                if ext in ["pdf", "docx"]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(payload)
                        tmp.flush() # Ensure data is written to disk
                    try:
                        extracted = extract_text_from_file(tmp.name, ext)
                        if extracted.strip():
                            extracted_parts.append(f"--- Attachment: {filename} ---\n{extracted}")
                    except Exception as e:
                        logger.warning(f"Failed to extract from attachment {filename}: {e}")
                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name) # Clean up temp file

            # Then process body parts if not an attachment
            elif not disposition: # No content-disposition usually indicates a body part
                if content_type == "text/plain" and payload:
                    extracted_parts.append(payload.decode(errors="ignore").strip())

                elif content_type == "text/html" and payload:
                    try:
                        html = payload.decode(errors="ignore")
                        soup = BeautifulSoup(html, "html.parser")
                        text = soup.get_text(separator="\n")
                        if text.strip():
                            extracted_parts.append(text.strip())
                    except Exception as e:
                        logger.warning(f"HTML parsing failed for email part ({content_type}): {e}")
    except Exception as e:
        logger.error(f"Error processing EML file '{path}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing EML file: {e}")

    final_text = "\n\n".join(part for part in extracted_parts if part.strip())
    if not final_text.strip():
        # Raise an exception if no readable content is found, as it's an issue for Q&A
        raise HTTPException(status_code=400, detail="No readable content found in email (body or supported attachments).")
    return final_text

def extract_text_from_url(url: str) -> str:
    """Fetches document from URL and extracts text based on extension."""
    logger.info(f"Attempting to fetch URL: {url}")
    try:
        response = requests.get(url, timeout=25) # Increased timeout slightly
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        logger.info(f"Successfully fetched URL: {url} with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch document from URL '{url}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL '{url}': {e}. Please check URL validity and network access.")
    
    ext = url.split('.')[-1].split('?')[0].lower()
    
    # Process EML in memory for efficiency if no attachments need temp files
    if ext == "eml":
        # For EML, write to temp file only to pass path to extract_text_from_eml
        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        try:
            return extract_text_from_eml(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path) # Clean up temp file
    else: # For PDF and DOCX, use temporary file as before
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        try:
            return extract_text_from_file(tmp_path, ext)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path) # Ensure temporary file is always deleted
    
    # This line should ideally be unreachable if all conditions are covered.
    logger.error(f"Unsupported file type encountered after all checks: {ext} for URL: {url}")
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: pdf, docx, eml.")

# ========================== NLP + GEMINI ==========================
def chunk_text(text: str, size=300, overlap=50) -> List[str]:
    """Chunks text into smaller pieces with overlap."""
    if not text:
        return []
    words = text.split()
    chunks = []
    if size <= overlap:
        logger.warning(f"chunk_size ({size}) was <= overlap ({overlap}). Adjusted chunk_size to {overlap + 1}.")
        size = overlap + 1

    for i in range(0, len(words), size - overlap):
        chunk = words[i:i + size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def build_index(chunks: List[str]):
    """Builds a FAISS index from text chunks."""
    global index, id_to_chunk_store, embedding_model # Use global model and index store
    
    index.reset() # Clear index for new document
    id_to_chunk_store.clear() # Clear mapped chunks

    if not chunks:
        logger.warning("No chunks provided to build FAISS index. Index will be empty.")
        return # No index or chunks if input is empty
    
    try:
        embeddings = embedding_model.encode(chunks, show_progress_bar=False) # Use global embedding_model
        index.add(np.array(embeddings).astype('float32')) # Ensure float32 for FAISS
        id_to_chunk_store.extend(chunks) # Store the chunks
        logger.info(f"FAISS index built with {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build search index: {e}")

def get_top_chunks(query: str, k=10) -> List[str]:
    """Retrieves top-k relevant chunks from the FAISS index."""
    global index, id_to_chunk_store, embedding_model # Use global model and index store
    
    if not id_to_chunk_store or index.ntotal == 0:
        logger.warning("FAISS index is empty or not built, cannot retrieve chunks.")
        return []
    
    try:
        query_vec = embedding_model.encode([query], show_progress_bar=False) # Use global embedding_model
        actual_top_k = min(k, index.ntotal)
        if actual_top_k == 0:
            return []

        D, I = index.search(np.array(query_vec).astype('float32'), actual_top_k)
        
        valid_indices = [i for i in I[0] if i != -1 and i < len(id_to_chunk_store)]
        return [id_to_chunk_store[i] for i in valid_indices]
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant document parts: {e}")

def answer_with_gemini(query: str, context: List[str]) -> str:
    """Generates an answer using Gemini based on provided context."""
    global gemini_model # Use global gemini_model

    if not context:
        return "No relevant information found in the document to answer the question."

    context_str = "\n\n".join(context)
    
    # Simple context length check (optional, but good for very long contexts)
    # Consider tokenizing for a more accurate check against Gemini's 1M token limit
    if len(context_str) > 500000: # Example: if context string is extremely long
        logger.warning("Context is very long, potentially impacting Gemini performance or hitting limits. Consider truncating.")
        # You might implement more sophisticated truncation here
        
    prompt = f"""
You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

Instructions:
1. Answer in complete sentences that **mirror the style and content of the original policy language**.
2. If a specific duration, condition, or percentage is mentioned, always include it.
3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
4. If no exact match exists, extract the most relevant and complete information available.
5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
6. Do not say "based on the context above" or "according to the document".
7. Be concise but **do not omit key legal terms or numbers**.
8. Do not generate an answer unless the information is supported in the context.

Now answer the following question:

Question: {query}

📄 Policy Context:
{context_str}

Respond concisely and clearly based only on the given context.
"""
    try:
        resp = gemini_model.generate_content(prompt) # Use global gemini_model
        return resp.text.strip() if resp and resp.text else "❗ No answer from Gemini"
    except Exception as e:
        logger.exception("Gemini failed during answer generation.") # Logs full traceback
        return f"❗ Gemini failed: {e}"

# ========================== AUTH ==========================
def verify_token(cred: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the bearer token against the TEAM_AUTH_TOKEN."""
    if cred.credentials != TEAM_AUTH_TOKEN:
        logger.warning("Unauthorized access attempt with invalid token.")
        raise HTTPException(status_code=403, detail="Unauthorized")

# ========================== ENDPOINT ==========================
@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """Main API endpoint to process documents and answer questions."""
    logger.info(f"Received query for document: {input_data.documents}")
    try:
        raw_text = extract_text_from_url(input_data.documents)
        logger.info(f"Text extracted. Length: {len(raw_text)} chars.")
        
        chunks = chunk_text(raw_text)
        logger.info(f"Text chunked. {len(chunks)} chunks generated.")
        
        build_index(chunks) # This will reset the global index and fill it for the current request
        logger.info("FAISS index built for current document.")

        answers = []
        for q_num, q_text in enumerate(input_data.questions):
            logger.info(f"Processing question {q_num + 1}: '{q_text}'")
            top_chunks = get_top_chunks(q_text) # No need to pass index, chunks, model
            ans = answer_with_gemini(q_text, top_chunks)
            answers.append(ans)
            logger.info(f"Answer for Q{q_num + 1}: '{ans[:75]}...'")

        logger.info("All questions processed. Returning answers.")
        return {"answers": answers}
    except HTTPException as e:
        # Re-raise HTTPExceptions directly for FastAPI to handle and return to client
        logger.error(f"HTTPException encountered: {e.detail} (Status: {e.status_code})")
        raise e
    except Exception as e:
        # Catch any other unexpected errors, log them with traceback, and return a 500 error to the client
        logger.exception("An unhandled exception occurred during query processing.") # Logs full traceback
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during query processing. Error: {str(e)}. Please check server logs for details.")

# This block is for local development testing with `python main.py`
# On Render, the `startCommand` in render.yaml (or direct service config) handles `uvicorn`
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) # reload=True for local dev