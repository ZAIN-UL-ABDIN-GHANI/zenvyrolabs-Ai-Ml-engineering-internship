# RAG AI Intern Assignment (Total: 300 Points)

Welcome to your AI Internship Assignment!

In this project, you will be building an Advanced Retrieval-Augmented Generation (RAG) Assistant that can perfectly answer questions based on a massive PDF textbook, a Novel, and even a Manga/Comic Book!

We have provided a starter template for you. It includes a beautiful ChatGPT-style frontend UI (in `frontend/`) with 3 distinct Assistant Modes (Coding, Novel, and Manga) and a basic FastAPI backend (in `backend/`). 

## Setup Instructions

1. Install Python: Ensure you have Python 3.10+ installed.
2. Install Libraries: Open your terminal inside the `backend` folder and run:
   ```bash
   pip install -r requirements.txt
   ```
3. Choose an LLM: We have stripped the LLM out of `rag_engine.py`. You must choose which LLM to use (e.g., Google Gemini, OpenAI, Groq, or a Local Model like Ollama). 
4. Configure your LLM: In `backend/rag_engine.py`, find the `llm = None` placeholder and initialize your chosen LLM.
5. Start the Server: Run the following command:
   ```bash
   python main.py
   ```
6. Open the UI: Go to `http://localhost:8000` in your web browser.

---

## Your Assignment: Fix the 5 Major Bugs (300 Points Total)

The starter code we provided contains a "Naive" RAG implementation. If you upload a book (which generates the local `chroma_db` database) and ask it questions, you will notice that it fails miserably on specific tasks. 

Your assignment is to modify the Python and JavaScript code to fix these 5 bugs:

### Bug 1: "Vector Blindness" & Absolute Precision (40 Points)
If you ask the AI, *"What happened on page 23?"* or *"What does page 34 line 5 say?"*, it will likely hallucinate or fail. 
* The Problem: The Semantic Search engine (`all-MiniLM-L6-v2`) only understands the *meaning* of words. It cannot "see" page numbers or exact locations because that data is hidden inside the Chunk Metadata, not in the actual text.
* Your Task: Modify `process_and_store_document()` in `rag_engine.py`. You must figure out how to dynamically inject absolute physical locations (like `[Source: PDF Viewer Page X]`) into the physical `page_content` of every chunk *before* it gets embedded into ChromaDB. The AI must be so precise it can answer exact locations perfectly.

### Bug 2: "Myopic Context" / Global Question Failure (40 Points)
If you ask the AI, *"Summarize the entire book"* or *"What is the best problem in the book?"*, it will fail or only summarize a tiny section.
* The Problem: The retriever is hardcoded to only pull `k=30` chunks (about 10 pages). It does not have enough context to read the whole book!
* Your Task: Modify `query_rag_system()` in `rag_engine.py`. You must write logic that detects if the user is asking a "Global" question. If they are, dynamically expand the `k` value or use a better retrieval strategy so your LLM can read massive sections of the textbook at once!

### Bug 3: "Amnesia" / No Conversation Memory (40 Points)
If you ask the AI a question, and then say *"What did I just ask you?"*, it will have no idea what you are talking about.
* The Problem: The frontend UI currently only sends your *latest* message to the API. Every time you hit send, the AI wakes up, answers, and forgets everything.
* Your Task: 
  1. Modify `app.js` in the frontend to maintain a `chatHistory` array. Send the entire history to the backend on every request.
  2. Modify `main.py` to accept the new history payload.
  3. Modify `rag_engine.py` to inject the conversation history directly into the LangChain Prompt Template so your LLM remembers the context!

### Bug 4: The Multimodal Manga OCR Challenge (100 Points - BOSS LEVEL)
If you click the "Manga / Comic" tab in the frontend and upload a Manga PDF, the AI will fail to extract any dialogue.
* The Problem: The `PyMuPDFLoader` we provided only extracts text from PDF text layers. Manga and Comics are essentially giant images (pictures of text).
* Your Task: The frontend already securely sends `book_type = "manga"` to the backend. You must modify `process_and_store_document()` to detect this. If it is a manga, you must bypass the standard loader and implement an **OCR (Optical Character Recognition)** library (like `pytesseract` or `EasyOCR`) or use a Vision AI API (like Gemini Flash) to extract the dialogue directly from the images before storing them in ChromaDB!

### Final Requirement: 1-on-1 Live Evaluation & Architecture Interview (80 Points)
You will have a live 1-on-1 evaluation meeting. You will share your screen, and you must successfully demonstrate that your AI can solve all of the problems above live! You must:
1. Upload your specific assigned textbook (generating the `chroma_db`).
2. Ask it a highly precise location question (e.g., "What does page 34 line 5 say?") to prove Bug 1 is fixed.
3. Ask a massive global summary question to prove Bug 2 is fixed.
4. Ask a follow-up memory question to prove Bug 3 is fixed.
5. Upload a Manga/Comic and ask it about a character's dialogue to prove your OCR pipeline (Bug 4) works!
6. **(Architecture Interview):** During the session, you will be verbally asked how you would scale your AI to handle a 500-page book without crashing free LLM APIs (Rate Limiting strategies). Be prepared to defend your logic!

> GRADING RULE: During the 1-on-1 evaluation, we will test each of the problems listed above to ensure everything is working correctly. Please note that any specific feature or bug fix that fails to work during this live screen-share session will receive 0 points for that section.
