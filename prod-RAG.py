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
import boto3
import pickle
import io
from dotenv import load_dotenv

load_dotenv()

class S3Handler:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        self.bucket_name = st.secrets["S3_BUCKET_NAME"]

    def upload_file(self, data, filename):
        try:
            # Convert data to bytes if it isn't already
            if not isinstance(data, bytes):
                data = pickle.dumps(data)
            
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=data
            )
            return True
        except Exception as e:
            st.error(f"Error uploading to S3: {str(e)}")
            return False

    def download_file(self, filename):
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=filename)
            return response['Body'].read()
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            st.error(f"Error downloading from S3: {str(e)}")
            return None

class VectorStore:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.s3 = S3Handler()
        
        # Try to load existing index and texts from S3
        index_data = self.s3.download_file('faiss_index.idx')
        texts_data = self.s3.download_file('stored_texts.pkl')
        
        if index_data and texts_data:
            # Load index from memory
            index_buffer = io.BytesIO(index_data)
            self.index = faiss.read_index_binary(index_buffer)
            self.texts = pickle.loads(texts_data)
        else:
            self.index = None
            self.texts = []
    
    def save_to_s3(self):
        if self.index:
            # Save index to binary buffer
            index_buffer = io.BytesIO()
            faiss.write_index_binary(self.index, index_buffer)
            self.s3.upload_file(index_buffer.getvalue(), 'faiss_index.idx')
            
            # Save texts
            self.s3.upload_file(self.texts, 'stored_texts.pkl')
    
    def add_texts(self, texts: List[str]):
        chunks = []
        for text in texts:
            chunks.extend(self.chunk_text(text))
        
        self.texts.extend(chunks)
        embeddings = self.encoder.encode(chunks, convert_to_tensor=True)
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.cpu().numpy())
        self.save_to_s3()  # Save after adding new texts
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.index or self.index.ntotal == 0:
            return [("No knowledge available. Please upload documents first.", 0.0)]
        
        query_vector = self.encoder.encode([query], convert_to_tensor=True)
        distances, indices = self.index.search(query_vector.cpu().numpy(), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(dist)))
        return results

class RAGApplication:
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.s3 = S3Handler()
    
    def process_file(self, uploaded_file) -> str:
        file_extension = os.path.splitext(uploaded_file.name.lower())[1]
        
        if file_extension not in {'.txt', '.pdf', '.docx'}:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            
            if file_extension == '.txt':
                text = uploaded_file.getvalue().decode('utf-8')
            elif file_extension == '.pdf':
                pdf_reader = PyPDF2.PdfReader(tmp_file.name)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            elif file_extension == '.docx':
                doc = docx.Document(tmp_file.name)
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            
            # Store the processed file in S3
            file_key = f"processed_files/{uploaded_file.name}"
            self.s3.upload_file(text.encode('utf-8'), file_key)
            
            return text

    def process_query(self, query: str, k: int = 3) -> str:
        relevant_docs = self.vector_store.similarity_search(query, k)
        context = "\n".join(f"Document {i+1}:\n{doc[0]}\n" 
                          for i, doc in enumerate(relevant_docs))
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Provide clear and concise answers based on the context provided. If there are any <think></think> tags in the response, remove them and their contents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        completion = self.client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages,
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True
        )
        
        return "".join(chunk.choices[0].delta.content or "" 
                      for chunk in completion if chunk.choices[0].delta.content)

def main():
    st.set_page_config(page_title="RAG System", layout="wide")
    st.title("RAG System")
    
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = RAGApplication()
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                try:
                    text = st.session_state.rag_app.process_file(file)
                    st.session_state.rag_app.vector_store.add_texts([text])
                    st.success(f"Successfully processed {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
    
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.rag_app.process_query(query)
                    st.write("Answer:", response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()