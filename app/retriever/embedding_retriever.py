# app/retriever/embedding_retriever.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLRetriever:
    """
    Retrieval system using embeddings for SHL assessments
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with an embedding model
        """
        self.model = SentenceTransformer(model_name)
        self.assessments = []
        self.embeddings = None
        self.index = None
        self.tfidf = TfidfVectorizer(max_features=1000)
        
    def prepare_documents(self, assessments: List[Dict]) -> List[str]:
        """
        Create searchable text from assessment data
        """
        documents = []
        for assessment in assessments:
            # Combine relevant fields for search
            text_parts = [
                assessment.get('name', ''),
                assessment.get('description', ''),
                ' '.join(assessment.get('test_type', [])),
                assessment.get('full_details', {}).get('long_description', ''),
                ' '.join(assessment.get('full_details', {}).get('skills_assessed', [])),
            ]
            
            # Filter out empty strings and join
            doc = ' '.join([part for part in text_parts if part])
            documents.append(doc)
            
        return documents
    
    def build_index(self, assessments: List[Dict]):
        """
        Build FAISS index from assessment embeddings
        """
        self.assessments = assessments
        documents = self.prepare_documents(assessments)
        
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k assessments for a query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.assessments):
                results.append((self.assessments[idx], float(score)))
                
        return results
    
    def hybrid_retrieve(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Tuple[Dict, float]]:
        """
        Hybrid retrieval combining embedding similarity with keyword matching
        """
        # Get embedding-based results
        embed_results = self.retrieve(query, k=k*2)
        
        # Simple keyword matching for reranking
        query_terms = set(query.lower().split())
        reranked = []
        
        for assessment, score in embed_results:
            # Boost score based on keyword matches
            text = self.prepare_documents([assessment])[0].lower()
            matches = sum(1 for term in query_terms if term in text)
            keyword_score = matches / len(query_terms) if query_terms else 0
            
            # Combine scores
            combined_score = alpha * score + (1 - alpha) * keyword_score
            reranked.append((assessment, combined_score))
        
        # Sort by combined score and return top-k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]
    
    def save(self, path: str):
        """
        Save retriever state
        """
        os.makedirs(path, exist_ok=True)
        
        # Save assessments
        with open(f"{path}/assessments.json", 'w') as f:
            json.dump(self.assessments, f)
        
        # Save embeddings and index
        if self.embeddings is not None:
            np.save(f"{path}/embeddings.npy", self.embeddings)
        
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/index.faiss")
            
        logger.info(f"Retriever saved to {path}")
    
    def load(self, path: str):
        """
        Load retriever state
        """
        # Load assessments
        with open(f"{path}/assessments.json", 'r') as f:
            self.assessments = json.load(f)
        
        # Load embeddings
        embeddings_path = f"{path}/embeddings.npy"
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        # Load index
        index_path = f"{path}/index.faiss"
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        logger.info(f"Retriever loaded from {path}")