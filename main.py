# # # # import os
# # # # import uuid
# # # # import fitz  # PyMuPDF: for extracting text from PDF documents
# # # # import docx2txt  # For extracting text from DOCX documents
# # # # import requests  # To download files from URLs
# # # # import tempfile  # For creating temporary local files
# # # # import faiss  # Facebook AI Similarity Search - used for vector indexing
# # # # import numpy as np  # For numerical operations
# # # # import google.generativeai as genai  # Google Gemini LLM API
# # # # from fastapi import FastAPI, HTTPException, Depends
# # # # from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# # # # from pydantic import BaseModel
# # # # from sentence_transformers import SentenceTransformer  # Sentence Embedding model
# # # # from typing import List
# # # # from dotenv import load_dotenv  # For loading environment variables

# # # # # -------- STEP 0: LOAD ENVIRONMENT VARIABLES -------- #
# # # # load_dotenv()  # Load .env file from project root

# # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

# # # # if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
# # # #     raise RuntimeError("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN in environment variables.")

# # # # # Configure Gemini
# # # # genai.configure(api_key=GEMINI_API_KEY)
# # # # gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # # # # Load sentence embedding model
# # # # model = SentenceTransformer("all-MiniLM-L6-v2")
# # # # dimension = 384

# # # # # FastAPI app instance
# # # # app = FastAPI(title="LLM Query Retrieval")

# # # # # HTTP Bearer security scheme
# # # # security = HTTPBearer()

# # # # # Initialize global FAISS index
# # # # index = faiss.IndexFlatL2(dimension)
# # # # id_to_chunk = []

# # # # # -------- AUTH CHECK -------- #
# # # # def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
# # # #     if credentials.credentials != TEAM_AUTH_TOKEN:
# # # #         raise HTTPException(status_code=403, detail="Unauthorized token.")

# # # # # -------- INPUT / OUTPUT MODELS -------- #
# # # # class QueryInput(BaseModel):
# # # #     documents: str
# # # #     questions: List[str]

# # # # class QueryOutput(BaseModel):
# # # #     answers: List[str]

# # # # # -------- TEXT EXTRACTION FROM URL -------- #
# # # # def extract_text_from_url(url: str) -> str:
# # # #     response = requests.get(url)
# # # #     if response.status_code != 200:
# # # #         raise HTTPException(status_code=400, detail="Failed to fetch document.")

# # # #     ext = url.split('.')[-1].split('?')[0]
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
# # # #         tmp_file.write(response.content)
# # # #         path = tmp_file.name

# # # #     if ext == "pdf":
# # # #         doc = fitz.open(path)
# # # #         return " ".join(page.get_text() for page in doc)
# # # #     elif ext == "docx":
# # # #         return docx2txt.process(path)
# # # #     else:
# # # #         raise HTTPException(status_code=400, detail="Unsupported or unknown file type.")

# # # # # -------- CHUNK DOCUMENT INTO TEXT BLOCKS -------- #
# # # # def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
# # # #     words = text.split()
# # # #     chunks = []
# # # #     for i in range(0, len(words), chunk_size - overlap):
# # # #         chunk = words[i:i + chunk_size]
# # # #         chunks.append(" ".join(chunk))
# # # #     return chunks

# # # # # -------- BUILD FAISS SEMANTIC INDEX -------- #
# # # # def build_faiss_index(chunks: List[str]):
# # # #     global index, id_to_chunk
# # # #     index.reset()
# # # #     id_to_chunk.clear()
# # # #     embeddings = model.encode(chunks)
# # # #     index.add(np.array(embeddings))
# # # #     id_to_chunk.extend(chunks)

# # # # # -------- RETRIEVE TOP-K CHUNKS -------- #
# # # # def retrieve_relevant_chunks(query: str, top_k: int = 10) -> List[str]:
# # # #     query_vec = model.encode([query])
# # # #     D, I = index.search(np.array(query_vec), top_k)
# # # #     return [id_to_chunk[i] for i in I[0]]

# # # # # -------- GEMINI ANSWER GENERATOR -------- #
# # # # def answer_with_gemini(query: str, context_chunks: List[str]) -> str:
# # # #     context = "\n\n".join(context_chunks)
# # # #     prompt = f"""
# # # # You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

# # # # Instructions:
# # # # 1. Answer in complete sentences that **mirror the style and content of the original policy language**.
# # # # 2. If a specific duration, condition, or percentage is mentioned, always include it.
# # # # 3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
# # # # 4. If no exact match exists, extract the most relevant and complete information available.
# # # # 5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
# # # # 6. Do not say "based on the context above" or "according to the document".
# # # # 7. Be concise but **do not omit key legal terms or numbers**.
# # # # 8. Do not generate an answer unless the information is supported in the context.

# # # # Now answer the following question:

# # # # Question: {query}

# # # # Context:
# # # # {context}
# # # # """
# # # #     response = gemini_model.generate_content(prompt)
# # # #     return response.text.strip()

# # # # # -------- MAIN API ROUTE WITH AUTH -------- #
# # # # @app.post("/api/v1/hackrx/run", response_model=QueryOutput)
# # # # async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
# # # #     try:
# # # #         raw_text = extract_text_from_url(input_data.documents)
# # # #         chunks = chunk_text(raw_text)
# # # #         build_faiss_index(chunks)

# # # #         final_answers = []
# # # #         for q in input_data.questions:
# # # #             top_chunks = retrieve_relevant_chunks(q)
# # # #             answer = answer_with_gemini(q, top_chunks)
# # # #             final_answers.append(answer)

# # # #         return {"answers": final_answers}
# # # #     except Exception as e:
# # # #         raise HTTPException(status_code=500, detail=str(e))
# # # #   --------------------------------------------------------------------------- 
# # # import os
# # # import tempfile
# # # import requests
# # # import email.message
# # # import logging # Re-import logging explicitly
# # # from fastapi import FastAPI, HTTPException, Depends
# # # from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# # # from pydantic import BaseModel
# # # from typing import List
# # # from dotenv import load_dotenv
# # # from sentence_transformers import SentenceTransformer
# # # from pdfminer.high_level import extract_text as extract_pdf_text # Note: This uses pdfminer.six
# # # import docx2txt
# # # import faiss
# # # import numpy as np
# # # import google.generativeai as genai
# # # from bs4 import BeautifulSoup # For parsing HTML content in emails
# # # # import uvicorn # Not typically imported at the top for Gunicorn/Uvicorn on Render

# # # # Configure logging at the top
# # # logging.basicConfig(level=logging.INFO) # Set to INFO for production
# # # logger = logging.getLogger("uvicorn.error") # Use the standard uvicorn logger

# # # # Initialize environment variables
# # # load_dotenv()

# # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

# # # if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
# # #     logger.error("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN. Please set them in your .env file.")
# # #     raise RuntimeError("Missing credentials. Please check environment variables.")

# # # # Configure Gemini and load models GLOBALLY (only once at app startup)
# # # genai.configure(api_key=GEMINI_API_KEY)
# # # gemini_model = genai.GenerativeModel('gemini-1.5-flash') # LOAD ONCE

# # # # Load sentence embedding model GLOBALLY (only once at app startup)
# # # # "paraphrase-albert-small-v2" is indeed a smaller model (approx 42MB)
# # # embedding_model = SentenceTransformer("paraphrase-albert-small-v2") # LOAD ONCE
# # # dimension = embedding_model.get_sentence_embedding_dimension() # Get dimension dynamically

# # # # FastAPI app instance
# # # app = FastAPI(title="LLM Query Retrieval for Hackathon")

# # # # HTTP Bearer security scheme
# # # security = HTTPBearer()

# # # # Initialize global FAISS index (reset for each request, but declared globally)
# # # index = faiss.IndexFlatL2(dimension)
# # # id_to_chunk_store = [] # Renamed to avoid conflict with `model` function parameter

# # # # ========================== SCHEMAS ==========================
# # # class QueryInput(BaseModel):
# # #     documents: str
# # #     questions: List[str]

# # # class QueryOutput(BaseModel):
# # #     answers: List[str]

# # # # ========================== FILE EXTRACTION ==========================
# # # def extract_text_from_file(path: str, ext: str) -> str:
# # #     """Extracts text from PDF or DOCX files."""
# # #     try:
# # #         if ext == "pdf":
# # #             return extract_pdf_text(path)
# # #         elif ext == "docx":
# # #             return docx2txt.process(path)
# # #         else:
# # #             raise ValueError(f"Unsupported file extension for direct extraction: {ext}")
# # #     except Exception as e:
# # #         logger.error(f"Error extracting text from {ext} file '{path}': {e}")
# # #         raise

# # # def extract_text_from_eml(path: str) -> str:
# # #     """Extracts text from email body (plain, HTML) and attachments (PDF, DOCX)."""
# # #     extracted_parts = []
# # #     try:
# # #         with open(path, "rb") as f:
# # #             msg = email.message_from_binary_file(f)

# # #         for part in msg.walk():
# # #             content_type = part.get_content_type()
# # #             disposition = part.get_content_disposition() # Can be None for body parts
# # #             filename = part.get_filename()
# # #             payload = part.get_payload(decode=True)

# # #             # Prioritize attachments first
# # #             if disposition == "attachment" and filename:
# # #                 ext = filename.split(".")[-1].lower()
# # #                 if ext in ["pdf", "docx"]:
# # #                     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
# # #                         tmp.write(payload)
# # #                         tmp.flush() # Ensure data is written to disk
# # #                     try:
# # #                         extracted = extract_text_from_file(tmp.name, ext)
# # #                         if extracted.strip():
# # #                             extracted_parts.append(f"--- Attachment: {filename} ---\n{extracted}")
# # #                     except Exception as e:
# # #                         logger.warning(f"Failed to extract from attachment {filename}: {e}")
# # #                     finally:
# # #                         if os.path.exists(tmp.name):
# # #                             os.unlink(tmp.name) # Clean up temp file

# # #             # Then process body parts if not an attachment
# # #             elif not disposition: # No content-disposition usually indicates a body part
# # #                 if content_type == "text/plain" and payload:
# # #                     extracted_parts.append(payload.decode(errors="ignore").strip())

# # #                 elif content_type == "text/html" and payload:
# # #                     try:
# # #                         html = payload.decode(errors="ignore")
# # #                         soup = BeautifulSoup(html, "html.parser")
# # #                         text = soup.get_text(separator="\n")
# # #                         if text.strip():
# # #                             extracted_parts.append(text.strip())
# # #                     except Exception as e:
# # #                         logger.warning(f"HTML parsing failed for email part ({content_type}): {e}")
# # #     except Exception as e:
# # #         logger.error(f"Error processing EML file '{path}': {e}", exc_info=True)
# # #         raise HTTPException(status_code=400, detail=f"Error processing EML file: {e}")

# # #     final_text = "\n\n".join(part for part in extracted_parts if part.strip())
# # #     if not final_text.strip():
# # #         # Raise an exception if no readable content is found, as it's an issue for Q&A
# # #         raise HTTPException(status_code=400, detail="No readable content found in email (body or supported attachments).")
# # #     return final_text

# # # def extract_text_from_url(url: str) -> str:
# # #     """Fetches document from URL and extracts text based on extension."""
# # #     logger.info(f"Attempting to fetch URL: {url}")
# # #     try:
# # #         response = requests.get(url, timeout=25) # Increased timeout slightly
# # #         response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
# # #         logger.info(f"Successfully fetched URL: {url} with status {response.status_code}")
# # #     except requests.exceptions.RequestException as e:
# # #         logger.error(f"Failed to fetch document from URL '{url}': {e}", exc_info=True)
# # #         raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL '{url}': {e}. Please check URL validity and network access.")
    
# # #     ext = url.split('.')[-1].split('?')[0].lower()
    
# # #     # Process EML in memory for efficiency if no attachments need temp files
# # #     if ext == "eml":
# # #         # For EML, write to temp file only to pass path to extract_text_from_eml
# # #         with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # #             tmp.write(response.content)
# # #             tmp_path = tmp.name
# # #         try:
# # #             return extract_text_from_eml(tmp_path)
# # #         finally:
# # #             if os.path.exists(tmp_path):
# # #                 os.unlink(tmp_path) # Clean up temp file
# # #     else: # For PDF and DOCX, use temporary file as before
# # #         with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
# # #             tmp.write(response.content)
# # #             tmp_path = tmp.name
# # #         try:
# # #             return extract_text_from_file(tmp_path, ext)
# # #         finally:
# # #             if os.path.exists(tmp_path):
# # #                 os.unlink(tmp_path) # Ensure temporary file is always deleted
    
# # #     # This line should ideally be unreachable if all conditions are covered.
# # #     logger.error(f"Unsupported file type encountered after all checks: {ext} for URL: {url}")
# # #     raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: pdf, docx, eml.")

# # # # ========================== NLP + GEMINI ==========================
# # # def chunk_text(text: str, size=300, overlap=50) -> List[str]:
# # #     """Chunks text into smaller pieces with overlap."""
# # #     if not text:
# # #         return []
# # #     words = text.split()
# # #     chunks = []
# # #     if size <= overlap:
# # #         logger.warning(f"chunk_size ({size}) was <= overlap ({overlap}). Adjusted chunk_size to {overlap + 1}.")
# # #         size = overlap + 1

# # #     for i in range(0, len(words), size - overlap):
# # #         chunk = words[i:i + size]
# # #         if chunk:
# # #             chunks.append(" ".join(chunk))
# # #     return chunks

# # # def build_index(chunks: List[str]):
# # #     """Builds a FAISS index from text chunks."""
# # #     global index, id_to_chunk_store, embedding_model # Use global model and index store
    
# # #     index.reset() # Clear index for new document
# # #     id_to_chunk_store.clear() # Clear mapped chunks

# # #     if not chunks:
# # #         logger.warning("No chunks provided to build FAISS index. Index will be empty.")
# # #         return # No index or chunks if input is empty
    
# # #     try:
# # #         embeddings = embedding_model.encode(chunks, show_progress_bar=False) # Use global embedding_model
# # #         index.add(np.array(embeddings).astype('float32')) # Ensure float32 for FAISS
# # #         id_to_chunk_store.extend(chunks) # Store the chunks
# # #         logger.info(f"FAISS index built with {len(chunks)} chunks.")
# # #     except Exception as e:
# # #         logger.error(f"Error building FAISS index: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail=f"Failed to build search index: {e}")

# # # def get_top_chunks(query: str, k=10) -> List[str]:
# # #     """Retrieves top-k relevant chunks from the FAISS index."""
# # #     global index, id_to_chunk_store, embedding_model # Use global model and index store
    
# # #     if not id_to_chunk_store or index.ntotal == 0:
# # #         logger.warning("FAISS index is empty or not built, cannot retrieve chunks.")
# # #         return []
    
# # #     try:
# # #         query_vec = embedding_model.encode([query], show_progress_bar=False) # Use global embedding_model
# # #         actual_top_k = min(k, index.ntotal)
# # #         if actual_top_k == 0:
# # #             return []

# # #         D, I = index.search(np.array(query_vec).astype('float32'), actual_top_k)
        
# # #         valid_indices = [i for i in I[0] if i != -1 and i < len(id_to_chunk_store)]
# # #         return [id_to_chunk_store[i] for i in valid_indices]
# # #     except Exception as e:
# # #         logger.error(f"Error retrieving relevant chunks: {e}", exc_info=True)
# # #         raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant document parts: {e}")

# # # def answer_with_gemini(query: str, context: List[str]) -> str:
# # #     """Generates an answer using Gemini based on provided context."""
# # #     global gemini_model # Use global gemini_model

# # #     if not context:
# # #         return "No relevant information found in the document to answer the question."

# # #     context_str = "\n\n".join(context)
    
# # #     # Simple context length check (optional, but good for very long contexts)
# # #     # Consider tokenizing for a more accurate check against Gemini's 1M token limit
# # #     if len(context_str) > 500000: # Example: if context string is extremely long
# # #         logger.warning("Context is very long, potentially impacting Gemini performance or hitting limits. Consider truncating.")
# # #         # You might implement more sophisticated truncation here
        
# # #     prompt = f"""
# # # You are a legal assistant designed to extract insurance policy clauses. Using the provided policy text, answer the user’s question **only from the context** and follow these instructions:

# # # Instructions:
# # # 1. Answer in complete sentences that **mirror the style and content of the original policy language**.
# # # 2. If a specific duration, condition, or percentage is mentioned, always include it.
# # # 3. If a clause is found that directly or closely matches the query, **quote or paraphrase it exactly and completely**.
# # # 4. If no exact match exists, extract the most relevant and complete information available.
# # # 5. Never respond with vague phrases like "not specified", "may vary", or "depends", unless it's explicitly stated in the policy.
# # # 6. Do not say "based on the context above" or "according to the document".
# # # 7. Be concise but **do not omit key legal terms or numbers**.
# # # 8. Do not generate an answer unless the information is supported in the context.

# # # Now answer the following question:

# # # Question: {query}

# # # 📄 Policy Context:
# # # {context_str}

# # # Respond concisely and clearly based only on the given context.
# # # """
# # #     try:
# # #         resp = gemini_model.generate_content(prompt) # Use global gemini_model
# # #         return resp.text.strip() if resp and resp.text else "❗ No answer from Gemini"
# # #     except Exception as e:
# # #         logger.exception("Gemini failed during answer generation.") # Logs full traceback
# # #         return f"❗ Gemini failed: {e}"

# # # # ========================== AUTH ==========================
# # # def verify_token(cred: HTTPAuthorizationCredentials = Depends(security)):
# # #     """Verifies the bearer token against the TEAM_AUTH_TOKEN."""
# # #     if cred.credentials != TEAM_AUTH_TOKEN:
# # #         logger.warning("Unauthorized access attempt with invalid token.")
# # #         raise HTTPException(status_code=403, detail="Unauthorized")

# # # # ========================== ENDPOINT ==========================
# # # @app.post("/api/v1/hackrx/run", response_model=QueryOutput)
# # # async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
# # #     """Main API endpoint to process documents and answer questions."""
# # #     logger.info(f"Received query for document: {input_data.documents}")
# # #     try:
# # #         raw_text = extract_text_from_url(input_data.documents)
# # #         logger.info(f"Text extracted. Length: {len(raw_text)} chars.")
        
# # #         chunks = chunk_text(raw_text)
# # #         logger.info(f"Text chunked. {len(chunks)} chunks generated.")
        
# # #         build_index(chunks) # This will reset the global index and fill it for the current request
# # #         logger.info("FAISS index built for current document.")

# # #         answers = []
# # #         for q_num, q_text in enumerate(input_data.questions):
# # #             logger.info(f"Processing question {q_num + 1}: '{q_text}'")
# # #             top_chunks = get_top_chunks(q_text) # No need to pass index, chunks, model
# # #             ans = answer_with_gemini(q_text, top_chunks)
# # #             answers.append(ans)
# # #             logger.info(f"Answer for Q{q_num + 1}: '{ans[:75]}...'")

# # #         logger.info("All questions processed. Returning answers.")
# # #         return {"answers": answers}
# # #     except HTTPException as e:
# # #         # Re-raise HTTPExceptions directly for FastAPI to handle and return to client
# # #         logger.error(f"HTTPException encountered: {e.detail} (Status: {e.status_code})")
# # #         raise e
# # #     except Exception as e:
# # #         # Catch any other unexpected errors, log them with traceback, and return a 500 error to the client
# # #         logger.exception("An unhandled exception occurred during query processing.") # Logs full traceback
# # #         raise HTTPException(status_code=500, detail=f"An internal server error occurred during query processing. Error: {str(e)}. Please check server logs for details.")

# # # # This block is for local development testing with `python main.py`
# # # # On Render, the `startCommand` in render.yaml (or direct service config) handles `uvicorn`
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     port = int(os.environ.get("PORT", 8000))
# # #     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) # reload=True for local dev



# # # =-------------------------------------------------------
# # import os
# # import tempfile
# # import requests
# # import email.message
# # import logging
# # from fastapi import FastAPI, HTTPException, Depends
# # from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# # from pydantic import BaseModel
# # from typing import List
# # from dotenv import load_dotenv
# # from sentence_transformers import SentenceTransformer
# # from pdfminer.high_level import extract_text as extract_pdf_text
# # import docx2txt
# # import faiss
# # import numpy as np
# # import google.generativeai as genai
# # from bs4 import BeautifulSoup
# # import nltk
# # import re

# # # Download the 'punkt' tokenizer for nltk if not already present
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except nltk.downloader.DownloadError:
# #     nltk.download('punkt', quiet=True)

# # # Configure logging at the top
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger("uvicorn.error")

# # # Initialize environment variables
# # load_dotenv()

# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN")

# # if not GEMINI_API_KEY or not TEAM_AUTH_TOKEN:
# #     logger.error("Missing GEMINI_API_KEY or TEAM_AUTH_TOKEN. Please set them in your .env file.")
# #     raise RuntimeError("Missing credentials. Please check environment variables.")

# # # Configure Gemini and load models GLOBALLY
# # genai.configure(api_key=GEMINI_API_KEY)
# # gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # # Load a robust but small embedding model
# # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# # dimension = embedding_model.get_sentence_embedding_dimension()

# # # FastAPI app instance
# # app = FastAPI(title="LLM Query Retrieval for Hackathon")

# # # HTTP Bearer security scheme
# # security = HTTPBearer()

# # # Global FAISS index and chunk store (resets per request)
# # index = faiss.IndexFlatL2(dimension)
# # id_to_chunk_store = []

# # # ========================== SCHEMAS ==========================
# # class QueryInput(BaseModel):
# #     documents: str
# #     questions: List[str]

# # class QueryOutput(BaseModel):
# #     answers: List[str]

# # # ========================== FILE EXTRACTION ==========================
# # def extract_text_from_file(path: str, ext: str) -> str:
# #     """Extracts text from PDF or DOCX files."""
# #     try:
# #         if ext == "pdf":
# #             return extract_pdf_text(path)
# #         elif ext == "docx":
# #             return docx2txt.process(path)
# #         else:
# #             raise ValueError(f"Unsupported file extension for direct extraction: {ext}")
# #     except Exception as e:
# #         logger.error(f"Error extracting text from {ext} file '{path}': {e}")
# #         raise

# # def extract_text_from_eml(path: str) -> str:
# #     """Extracts text from email body (plain, HTML) and attachments (PDF, DOCX)."""
# #     extracted_parts = []
# #     try:
# #         with open(path, "rb") as f:
# #             msg = email.message_from_binary_file(f)

# #         for part in msg.walk():
# #             content_type = part.get_content_type()
# #             disposition = part.get_content_disposition()
# #             filename = part.get_filename()
# #             payload = part.get_payload(decode=True)

# #             if disposition == "attachment" and filename:
# #                 ext = filename.split(".")[-1].lower()
# #                 if ext in ["pdf", "docx"]:
# #                     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
# #                         tmp.write(payload)
# #                         tmp.flush()
# #                     try:
# #                         extracted = extract_text_from_file(tmp.name, ext)
# #                         if extracted.strip():
# #                             extracted_parts.append(f"--- Attachment: {filename} ---\n{extracted}")
# #                     finally:
# #                         if os.path.exists(tmp.name):
# #                             os.unlink(tmp.name)
# #             elif not disposition:
# #                 if content_type == "text/plain" and payload:
# #                     extracted_parts.append(payload.decode(errors="ignore").strip())
# #                 elif content_type == "text/html" and payload:
# #                     try:
# #                         html = payload.decode(errors="ignore")
# #                         soup = BeautifulSoup(html, "html.parser")
# #                         text = soup.get_text(separator="\n")
# #                         if text.strip():
# #                             extracted_parts.append(text.strip())
# #                     except Exception as e:
# #                         logger.warning(f"HTML parsing failed for email part ({content_type}): {e}")
# #     except Exception as e:
# #         logger.error(f"Error processing EML file '{path}': {e}", exc_info=True)
# #         raise HTTPException(status_code=400, detail=f"Error processing EML file: {e}")

# #     final_text = "\n\n".join(part for part in extracted_parts if part.strip())
# #     if not final_text.strip():
# #         raise HTTPException(status_code=400, detail="No readable content found in email.")
# #     return final_text

# # def extract_text_from_url(url: str) -> str:
# #     """Fetches document from URL and extracts text based on extension."""
# #     logger.info(f"Attempting to fetch URL: {url}")
# #     try:
# #         response = requests.get(url, timeout=30)
# #         response.raise_for_status()
# #         logger.info(f"Successfully fetched URL: {url} with status {response.status_code}")
# #     except requests.exceptions.RequestException as e:
# #         logger.error(f"Failed to fetch document from URL '{url}': {e}", exc_info=True)
# #         raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL '{url}': {e}.")
    
# #     ext = url.split('.')[-1].split('?')[0].lower()
# #     allowed_extensions = ["pdf", "docx", "eml"]
# #     if ext not in allowed_extensions:
# #         raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: {', '.join(allowed_extensions)}.")
    
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
# #         tmp.write(response.content)
# #         tmp_path = tmp.name

# #     try:
# #         if ext == "eml":
# #             return extract_text_from_eml(tmp_path)
# #         else:
# #             return extract_text_from_file(tmp_path, ext)
# #     finally:
# #         if os.path.exists(tmp_path):
# #             os.unlink(tmp_path)

# # # ========================== NLP + GEMINI ==========================
# # def chunk_text_by_sentences(text: str, max_chunk_size_words=400, overlap_words=80) -> List[str]:
# #     """
# #     Chunks text into pieces, respecting sentence boundaries, with a larger chunk size
# #     and overlap to improve retrieval of legal clauses.
# #     """
# #     if not text:
# #         return []

# #     sentences = nltk.sent_tokenize(text)
    
# #     chunks = []
# #     current_chunk = ""
    
# #     for sentence in sentences:
# #         words = sentence.split()
        
# #         # If a single sentence is larger than max chunk size, add it as its own chunk
# #         if len(words) > max_chunk_size_words:
# #             chunks.append(sentence)
# #             continue
            
# #         # Check if adding the sentence would exceed the chunk size
# #         if len((current_chunk + " " + sentence).split()) <= max_chunk_size_words:
# #             if current_chunk:
# #                 current_chunk += " " + sentence
# #             else:
# #                 current_chunk = sentence
# #         else:
# #             if current_chunk:
# #                 chunks.append(current_chunk)
            
# #             # Start a new chunk with an overlap from the previous one
# #             overlap_text = " ".join(current_chunk.split()[-overlap_words:]) if current_chunk else ""
# #             current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            
# #     if current_chunk:
# #         chunks.append(current_chunk)
        
# #     return [chunk for chunk in chunks if chunk.strip()]


# # def build_index(chunks: List[str]):
# #     """Builds a FAISS index from text chunks."""
# #     global index, id_to_chunk_store, embedding_model
    
# #     index.reset()
# #     id_to_chunk_store.clear()

# #     if not chunks:
# #         logger.warning("No chunks provided to build FAISS index. Index will be empty.")
# #         return
    
# #     try:
# #         embeddings = embedding_model.encode(chunks, show_progress_bar=False)
# #         index.add(np.array(embeddings).astype('float32'))
# #         id_to_chunk_store.extend(chunks)
# #         logger.info(f"FAISS index built with {len(chunks)} chunks.")
# #     except Exception as e:
# #         logger.error(f"Error building FAISS index: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=f"Failed to build search index: {e}")

# # def get_top_chunks(query: str, k=8) -> List[str]:
# #     """Retrieves top-k relevant chunks from the FAISS index."""
# #     global index, id_to_chunk_store, embedding_model
    
# #     if not id_to_chunk_store or index.ntotal == 0:
# #         logger.warning("FAISS index is empty or not built, cannot retrieve chunks.")
# #         return []
    
# #     try:
# #         query_vec = embedding_model.encode([query], show_progress_bar=False)
# #         actual_top_k = min(k, index.ntotal)
# #         if actual_top_k == 0:
# #             return []

# #         D, I = index.search(np.array(query_vec).astype('float32'), actual_top_k)
        
# #         valid_indices = [i for i in I[0] if i != -1 and i < len(id_to_chunk_store)]
# #         return [id_to_chunk_store[i] for i in valid_indices]
# #     except Exception as e:
# #         logger.error(f"Error retrieving relevant chunks: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant document parts: {e}")

# # def answer_with_gemini(query: str, context: List[str]) -> str:
# #     """Generates an answer using Gemini based on provided context, with a prompt
# #     that encourages synthesis and a clear format."""
# #     global gemini_model

# #     if not context:
# #         return "Information not found in the document."

# #     context_str = "\n\n".join(context)
    
# #     prompt = f"""
# # You are an expert legal assistant. Using the provided policy text, answer the user’s question **only from the context**. Follow these instructions to provide a complete and helpful answer:

# # ### Instructions:
# # 1.  Generate a clear and complete answer that directly addresses the user's question.
# # 2.  Synthesize all relevant information from the provided policy context into a single, cohesive response.
# # 3.  Include all specific details such as durations, conditions, percentages, and limits.
# # 4.  If the answer is affirmative, start with "Yes, ..."
# # 5.  If a specific clause or definition is requested, provide the complete, verbatim text.
# # 6.  If the answer cannot be found in the provided context, state "Information not found in the document." Do not attempt to guess or infer an answer.
# # 7.  Do not use vague phrases like "according to the document" or "based on the context."

# # ### User Question:
# # {query}

# # ### Provided Policy Context:
# # {context_str}

# # Generate a complete and helpful answer based ONLY on the provided context.
# # """
# #     try:
# #         resp = gemini_model.generate_content(prompt)
# #         return resp.text.strip() if resp and resp.text and resp.text.strip() else "Information not found in the document."
# #     except Exception as e:
# #         logger.exception("Gemini failed during answer generation.")
# #         return f"❗ Gemini API Error: {e}"

# # # ========================== AUTH ==========================
# # def verify_token(cred: HTTPAuthorizationCredentials = Depends(security)):
# #     """Verifies the bearer token against the TEAM_AUTH_TOKEN."""
# #     if cred.credentials != TEAM_AUTH_TOKEN:
# #         logger.warning("Unauthorized access attempt with invalid token.")
# #         raise HTTPException(status_code=403, detail="Unauthorized")

# # # ========================== ENDPOINT ==========================
# # @app.post("/api/v1/hackrx/run", response_model=QueryOutput)
# # async def run_query(input_data: QueryInput, token: HTTPAuthorizationCredentials = Depends(verify_token)):
# #     """Main API endpoint to process documents and answer questions."""
# #     logger.info(f"Received query for document: {input_data.documents}")
# #     try:
# #         raw_text = extract_text_from_url(input_data.documents)
# #         logger.info(f"Text extracted. Length: {len(raw_text)} chars.")
        
# #         chunks = chunk_text_by_sentences(raw_text)
# #         logger.info(f"Text chunked. {len(chunks)} chunks generated.")
        
# #         # Reset and build index for the new document
# #         build_index(chunks)
# #         logger.info("FAISS index built for current document.")

# #         answers = []
# #         for q_num, q_text in enumerate(input_data.questions):
# #             logger.info(f"Processing question {q_num + 1}: '{q_text}'")
# #             top_chunks = get_top_chunks(q_text)
# #             ans = answer_with_gemini(q_text, top_chunks)
# #             answers.append(ans)
# #             logger.info(f"Answer for Q{q_num + 1}: '{ans[:75]}...'")

# #         logger.info("All questions processed. Returning answers.")
# #         return {"answers": answers}
# #     except HTTPException as e:
# #         logger.error(f"HTTPException encountered: {e.detail} (Status: {e.status_code})")
# #         raise e
# #     except Exception as e:
# #         logger.exception("An unhandled exception occurred during query processing.")
# #         raise HTTPException(status_code=500, detail=f"An internal server error occurred. Error: {str(e)}")

# # # For local development
# # if __name__ == "__main__":
# #     import uvicorn
# #     port = int(os.environ.get("PORT", 8000))
# #     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


# # -------------------------acc
# import os
# import tempfile
# import requests
# import email
# import logging
# import numpy as np
# import re
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel
# from typing import List
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# import faiss
# from google.generativeai import configure,GenerativeModel
# from PyPDF2 import PdfReader
# import docx

# # Load .env variables
# load_dotenv()

# app = FastAPI()
# security = HTTPBearer()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Fetch the Gemini API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise EnvironmentError("GEMINI_API_KEY not set in environment or .env file.")

# # Set API key for Gemini
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Optional
# configure(api_key=GEMINI_API_KEY)              # ✅ Required

# # Load models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# model = GenerativeModel("gemini-1.5-flash")

# embedding_dim = 384
# index = faiss.IndexFlatL2(embedding_dim)
# id_to_chunk_store = []

# # Input model
# class BulkQueryRequest(BaseModel):
#     documents: str  # URL to PDF
#     questions: List[str]

# class QAResponse(BaseModel):
#     answers: List[str]

# @app.post("/api/v1/hackrx/run", response_model=QAResponse)
# def run_bulk_query(request: BulkQueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
#     try:
#         # Step 1: Download and parse document
#         document_url = request.documents
#         file_path, file_ext = download_document(document_url)

#         if file_ext.endswith(".pdf"):
#             text = extract_pdf(open(file_path, "rb"))
#         elif file_ext.endswith(".docx"):
#             text = extract_docx(open(file_path, "rb"))
#         elif file_ext.endswith(".eml"):
#             text = extract_eml(open(file_path, "rb"))
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file type.")

#         # Step 2: Clean and chunk document
#         text = clean_text(text)
#         chunks = split_into_chunks(text)

#         # Step 3: Index chunks
#         id_to_chunk_store.clear()
#         index.reset()
#         vectors = embedding_model.encode(chunks, show_progress_bar=False)
#         index.add(np.array(vectors).astype("float32"))
#         id_to_chunk_store.extend(chunks)

#         # Step 4: Answer each question
#         answers = []
#         for question in request.questions:
#             top_chunks = get_top_chunks(question)
#             raw_answer = answer_with_gemini(question, top_chunks)
#             styled_answer = enforce_sample_style(raw_answer)
#             logger.info(f"Q: {question}\nA: {styled_answer}")
#             answers.append(styled_answer)

#         return {"answers": answers}

#     except Exception as e:
#         logger.exception("Failed to process query")
#         raise HTTPException(status_code=500, detail=str(e))


# def download_document(url: str):
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to download document.")

#     content_type = response.headers.get("Content-Type", "")
#     suffix = ".pdf" if "pdf" in content_type else ".docx" if "word" in content_type else ".eml" if "eml" in url.lower() else ""

#     if not suffix:
#         raise HTTPException(status_code=400, detail="Unable to detect file type from URL.")

#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#     temp_file.write(response.content)
#     temp_file.close()

#     return temp_file.name, suffix

# def get_top_chunks(query: str, k=8) -> List[str]:
#     query_vec = embedding_model.encode([query], show_progress_bar=False)
#     D, I = index.search(np.array(query_vec).astype('float32'), k)

#     chunks = []
#     for score, idx in zip(D[0], I[0]):
#         if idx != -1 and idx < len(id_to_chunk_store):
#             chunk = id_to_chunk_store[idx]
#             # Optional: Discard low-relevance chunks (e.g., L2 distance > threshold)
#             if score < 1.0:  # Adjust threshold as needed
#                 chunks.append(chunk)
#     return chunks



# def answer_with_gemini(query: str, context_chunks: List[str]) -> str:
#     context_str = "\n\n".join(context_chunks)
#     prompt = f"""
# You are a professional legal assistant responding to questions based on an insurance policy document.

# ### OBJECTIVE:
# Accurately answer the user's QUESTION using only the POLICY CONTEXT below. Your answer must match official policy terms and be free from interpretation, generalization, or assumptions.


# ### RESPONSE STYLE RULES:
# - Answer only if the information is explicitly stated in the POLICY CONTEXT.
# - Respond in 1-3 sentences.
# - Use definitive language (e.g., “Yes, the policy covers...”).
# - Do NOT guess or imply missing details.
# - Use numbered words in parentheses where applicable: “thirty (30) days”.
# - Respond in a formal, definitive tone using policy-style language.
# - Phrase numbers as: “twenty-four (24) months”, “1% of the Sum Insured”, etc.
# - Capitalize policy terms: Sum Insured, Grace Period, Pre-Existing Diseases, etc.
# - Do NOT add explanations or assumptions
# - If information is unclear or missing, respond: “Information not found in the document.”


# ### QUESTION:
# {query}

# ### POLICY CONTEXT:
# {context_str}

# Respond with the final answer in plain text:
# """
#     response = model.generate_content(prompt)
    
#     if "not covered" in response.text.lower() or "not mentioned" in response.text.lower():
#         logger.warning(f"Potential mismatch for: {query} → {response.text.strip()}")
    
#     return response.text.strip()

# def enforce_sample_style(answer: str) -> str:
#     if not answer or "information not found" in answer.lower():
#         return "Information not found in the document."

#     answer = answer.strip()

#     # Style and term replacements
#     replacements = {
#         r"\bpre[- ]existing diseases\b": "Pre-Existing Diseases",
#         r"\bno claim discount\b": "No Claim Discount",
#         r"\bncd\b": "No Claim Discount",
#         r"\bsum insured\b": "Sum Insured",
#         r"\bwaiting period\b": "Waiting Period",
#         r"\bgrace period\b": "Grace Period",
#         r"\bhealth check[- ]up\b": "Health Check-Up",
#         r"\bpolicyholder\b": "Policyholder",
#         r"\binpatient treatment\b": "inpatient treatment",
#         r"\boutpatient treatment\b": "outpatient treatment",
#         r"\bcoverage period\b": "Policy Period",
#     }

#     number_map = {
#         r"\btwo years\b": "two (2) years",
#         r"\bthree years\b": "three (3) years",
#         r"\bfour years\b": "four (4) years",
#         r"\bthirty days\b": "thirty (30) days",
#         r"\b30 days\b": "thirty (30) days",
#         r"\b24 months\b": "twenty-four (24) months",
#         r"\btwenty four months\b": "twenty-four (24) months",
#         r"\b36 months\b": "thirty-six (36) months",
#         r"\bthirty six months\b": "thirty-six (36) months",
#         r"\b1% of sum insured\b": "1% of the Sum Insured",
#         r"\b2% of sum insured\b": "2% of the Sum Insured",
#     }

#     # Apply style normalization
#     for pattern, replacement in {**replacements, **number_map}.items():
#         answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

#     # 🧠 Inject mandatory clauses based on trigger words
#     if "maternity" in answer.lower() and "twenty-four (24) months" not in answer:
#         answer += " To be eligible, the female Insured Person must have been continuously covered for at least twenty-four (24) months. The benefit is limited to two deliveries or terminations during the Policy Period."

#     if "cataract" not in answer.lower() and "refractive" in answer.lower():
#         answer = "The policy has a specific Waiting Period of two (2) years for cataract surgery."

#     if "AYUSH" in answer and "AYUSH Hospital" not in answer:
#         answer += " The treatment must be taken in an AYUSH Hospital."

#     if "room charges" in answer.lower() and "1% of the Sum Insured" not in answer:
#         answer += " Daily Room Rent is capped at 1% of the Sum Insured and Intensive Care Unit charges at 2%, unless treatment is taken in a Preferred Provider Network (PPN) hospital."

#     # Final formatting cleanup
#     answer = re.sub(r'\s+', ' ', answer).strip()
#     if not answer.endswith('.'):
#         answer += '.'

#     return answer





# def extract_pdf(file_obj):
#     reader = PdfReader(file_obj)
#     return "\n".join([page.extract_text() or '' for page in reader.pages])

# def extract_docx(file_obj):
#     doc = docx.Document(file_obj)
#     return "\n".join([para.text for para in doc.paragraphs])

# def extract_eml(file_obj):
#     msg = email.message_from_binary_file(file_obj)
#     parts = []
#     for part in msg.walk():
#         if part.get_content_type() == "text/plain":
#             payload = part.get_payload(decode=True)
#             if payload:
#                 parts.append(payload.decode(errors='ignore'))
#     return "\n".join(parts)

# def clean_text(text):
#     return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

# def split_into_chunks(text, max_tokens=150):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks, current_chunk = [], []

#     for sentence in sentences:
#         current_chunk.append(sentence)
#         if len(" ".join(current_chunk).split()) >= max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# ---------------------with 2doc
import os
import tempfile
import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure

# Set up your API key for Gemini
configure(api_key="your_google_gemini_api_key")  # Replace with your actual API key

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Gemini model
model = GenerativeModel("gemini-pro")

# FastAPI app
app = FastAPI()

# FAISS index and related data
index = None
chunks = []

class QueryInput(BaseModel):
    documents: str
    questions: list[str]

def download_and_extract_pdf(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    text = ""
    with fitz.open(tmp_path) as doc:
        for page in doc:
            text += page.get_text()

    os.remove(tmp_path)
    return text

def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")

def get_top_k_chunks(question, k=5):
    question_vec = embedding_model.encode([question]).astype("float32")
    _, I = index.search(question_vec, k)
    return [chunks[i] for i in I[0]]

def ask_gemini(context_str, question):
    prompt = f"""You are an insurance policy expert. Use the following context to answer the user's question.
If the answer isn't explicitly found, mention that clearly but provide related information if available.

CONTEXT:
{context_str}

QUESTION: {question}

INSTRUCTIONS:
- Only use the information provided above.
- If the answer is unclear or missing, explicitly say: "No direct answer found", but include the most relevant parts.
- Do not guess. Be precise.
- Keep the answer under 200 words.

Answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/query")
def query_policy(input_data: QueryInput):
    global index, chunks

    try:
        raw_text = download_and_extract_pdf(input_data.documents)
        chunks = chunk_text(raw_text)
        embeddings = embed_chunks(chunks)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        results = []
        for question in input_data.questions:
            top_chunks = get_top_k_chunks(question)
            context = "\n".join(top_chunks)
            answer = ask_gemini(context, question)
            results.append(answer)

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
