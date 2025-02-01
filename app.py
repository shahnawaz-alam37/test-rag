import streamlit as st
import PyPDF2
import docx
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import torch
import os
import tempfile
from typing import List, Tuple
import textwrap
import json
import pickle
import boto3
from botocore.exceptions import ClientError
import io

GROQ_API_KEY =st.secrets["GROQ_API_KEY"]  # Replace with actual Groq API key
FAISS_INDEX_PATH = "faiss_index.idx"
TEXT_STORE_PATH = "stored_texts.pkl"

# Add AWS credentials
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
S3_BUCKET_NAME = "rag-test1-edusmart"

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.txt', '.pdf', '.docx'}
        
    def read_text_file(self, file) -> str:
        return file.read().decode('utf-8')
    
    def read_pdf_file(self, file) -> str:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def read_docx_file(self, file) -> str:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def process_file(self, uploaded_file) -> str:
        _, file_extension = os.path.splitext(uploaded_file.name.lower())
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            
            if file_extension == '.txt':
                return self.read_text_file(uploaded_file)
            elif file_extension == '.pdf':
                return self.read_pdf_file(tmp_file.name)
            elif file_extension == '.docx':
                return self.read_docx_file(tmp_file.name)

class VectorStore:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.index = None
        self.texts = []
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        # Load index and stored texts from S3 if available
        try:
            # First check if files exist
            try:
                self.s3_client.head_object(Bucket=S3_BUCKET_NAME, Key='stored_texts.pkl')
            except ClientError:
                print("No existing texts found in S3. Starting fresh.")
                return

            # Download stored texts first
            with io.BytesIO() as texts_buffer:
                self.s3_client.download_fileobj(S3_BUCKET_NAME, 'stored_texts.pkl', texts_buffer)
                texts_buffer.seek(0)
                if texts_buffer.getvalue():  # Check if buffer is not empty
                    self.texts = pickle.load(texts_buffer)
            
            # Create new index from texts
            if self.texts:
                embeddings = self.encoder.encode(self.texts)  # Remove convert_to_tensor=True
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings)  # Add numpy array directly
                
                try:
                    # Try to load existing index if available
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        self.s3_client.download_fileobj(S3_BUCKET_NAME, 'faiss_index.idx', tmp_file)
                        tmp_file.flush()
                        self.index = faiss.read_index(tmp_file.name)  # Changed from read_index_binary to read_index
                    os.unlink(tmp_file.name)
                except Exception as e:
                    print(f"Could not load existing index: {e}. Using newly created index.")
                    
        except Exception as e:
            print(f"Initialization error: {str(e)}. Starting fresh.")
            self.index = None
            self.texts = []

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def add_texts(self, texts: List[str]):
        chunks = []
        for text in texts:
            chunks.extend(self.chunk_text(text))
        
        self.texts.extend(chunks)
        embeddings = self.encoder.encode(chunks)  # Using numpy array directly
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings)
        
        # Save the updated index and texts to S3
        try:
            # Save FAISS index to S3
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                faiss.write_index(self.index, tmp_file.name)  # Changed from write_index_binary to write_index
                tmp_file.flush()
                with open(tmp_file.name, 'rb') as f:
                    self.s3_client.upload_fileobj(f, S3_BUCKET_NAME, 'faiss_index.idx')
            os.unlink(tmp_file.name)

            # Save stored texts to S3
            with io.BytesIO() as texts_buffer:
                pickle.dump(self.texts, texts_buffer)
                texts_buffer.seek(0)
                self.s3_client.upload_fileobj(texts_buffer, S3_BUCKET_NAME, 'stored_texts.pkl')
        except ClientError as e:
            print(f"Error saving to S3: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.index or self.index.ntotal == 0:
            return [("No knowledge available. Please upload documents first.", 0.0)]
        
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(dist)))
        
        return results

class RAGApplication:
    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("API key is required")
            
        # Initialize components only when needed
        self._vector_store = None
        self._doc_processor = None
        self._client = None
        self.api_key = api_key
    
    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    @property
    def doc_processor(self):
        if self._doc_processor is None:
            self._doc_processor = DocumentProcessor()
        return self._doc_processor
    
    @property
    def client(self):
        if self._client is None:
            # Initialize Groq client with minimal arguments
            self._client = Groq(api_key=self.api_key)
        return self._client

    def process_query(self, query: str, k: int = 3) -> str:
        try:
            relevant_docs = self.vector_store.similarity_search(query, k)
            
            context = "\n".join([f"Document {i+1}:\n{doc[0]}\n" 
                               for i, doc in enumerate(relevant_docs)])
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear and concise answers based on the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            # Updated Groq chat completion parameters
            completion = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Changed to a more reliable model
                messages=messages,
                temperature=0.7,
                max_tokens=2048,  # Changed from max_completion_tokens
                top_p=0.95,
                stream=False  # Changed to False for simpler handling
            )
            
            # Simplified response handling
            return completion.choices[0].message.content
            
        except Exception as e:
            st.error(f"Query processing error: {str(e)}")
            raise
    
    def generate_mind_map(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Prompt: {prompt}"}
        ]
        
        completion = self.client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages,
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
        )
        
        response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                response += content
        
        try:
            mind_map = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding mind map: {str(e)}")
        
        return mind_map

def main():
    try:
        # Use wider layout
        st.set_page_config(
            page_title="Advanced RAG System",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS to ensure inputs render properly
        st.markdown("""
            <style>
                .stTextInput > div > div > input {
                    width: 100%;
                }
                .stFileUploader > div > div {
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True)
        
        st.title("Advanced RAG System")
        
        # Initialize the RAG application with error catching
        if 'rag_app' not in st.session_state:
            try:
                st.session_state.rag_app = RAGApplication(api_key=GROQ_API_KEY)
                st.success("Application initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing application: {str(e)}")
                st.stop()
        
        # File upload section with error handling
        st.header("Document Upload")
        try:
            uploaded_files = st.file_uploader(
                "Upload your documents (PDF, DOCX, TXT)", 
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt'],
                key="file_uploader"  # Add a unique key
            )
        except Exception as e:
            st.error(f"Error with file uploader: {str(e)}")
            uploaded_files = None
        
        # Query section with error handling
        st.header("Ask Questions")
        try:
            query = st.text_input("Enter your question:", key="query_input")  # Add a unique key
        except Exception as e:
            st.error(f"Error with text input: {str(e)}")
            query = None
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    try:
                        text = st.session_state.rag_app.doc_processor.process_file(file)
                        st.session_state.rag_app.vector_store.add_texts([text])
                        st.success(f"Successfully processed {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
        
        if st.button("Get Answer"):
            if not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating response..."):
                    try:
                        response = st.session_state.rag_app.process_query(query)
                        st.write("Answer:", response)
                        st.code(response, language='text')
                        if st.button("Copy Answer"):
                            st.write("Answer copied to clipboard!")
                            st.experimental_set_query_params(answer=response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
        
        # # Mind map generation section
        # st.header("Generate Mind Map")
        # prompt = st.text_input("Enter a prompt for mind map:")
        
        # if st.button("Generate Mind Map"):
        #     if not prompt:
        #         st.warning("Please enter a prompt.")
        #     else:
        #         with st.spinner("Generating mind map..."):
        #             try:
        #                 mind_map = st.session_state.rag_app.generate_mind_map(prompt)
        #                 st.write("Mind Map:", mind_map)
        #             except Exception as e:
        #                 st.error(f"Error generating mind map: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
