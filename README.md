# Reader-Chatbot

Intelligent RAG Chatbot for Document Q&A


Project Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that transforms large uploaded documents (PDF, DOCX, CSV, TXT) into an interactive question-answering system. It is designed to efficiently understand and answer queries based on the content of complex manuals or documents.


Key features include:
	•	Semantic paragraph-based chunking: Splits documents into meaningful, contextually coherent chunks for optimal retrieval.
	•	Hybrid retrieval approach: Combines embedding-based semantic search (using SentenceTransformers and FAISS) with keyword-based search (BM25) to improve retrieval accuracy.
	•	Generative AI answer synthesis: Leverages Google Gemini generative models to produce precise and concise answers from the retrieved context.
	•	User-friendly Streamlit interface: Allows users to upload documents, ask questions, and view chat history seamlessly.
	•	Robust data cleaning and error handling: Ensures better text extraction and smooth user experience even with large manuals.
