import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import re
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLRecommender:
    def __init__(self, catalog_path='data/shl_catalog.csv'):
        """Initialize the recommender"""
        logger.info("Loading catalog...")
        self.catalog = pd.read_csv(catalog_path)
        
        # Handle test_type column which might be stored as string
        if 'test_type' in self.catalog.columns:
            self.catalog['test_type'] = self.catalog['test_type'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        logger.info(f"Loaded {len(self.catalog)} assessments")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or load embeddings
        self.embeddings = self.load_or_create_embeddings()
        
        # Create TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.create_tfidf_matrix()
        
    def create_assessment_text(self, row):
        """Create rich text for embedding"""
        parts = []
        
        # Add name
        if pd.notna(row.get('name')):
            parts.append(str(row['name']))
        
        # Add description
        if pd.notna(row.get('description')):
            parts.append(str(row['description']))
        
        # Add test types as text
        test_types = row.get('test_type', [])
        if isinstance(test_types, list):
            type_text = ' '.join(test_types)
            parts.append(f"Test types: {type_text}")
        
        # Add duration info
        duration = row.get('duration', 0)
        if pd.notna(duration) and duration > 0:
            parts.append(f"Duration: {int(duration)} minutes")
        
        # Add support info
        if pd.notna(row.get('adaptive_support')):
            parts.append(f"Adaptive: {row['adaptive_support']}")
        
        if pd.notna(row.get('remote_support')):
            parts.append(f"Remote: {row['remote_support']}")
        
        return ' '.join(parts)
    
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        try:
            embeddings = np.load('data/embeddings.npy')
            logger.info("Loaded existing embeddings")
            return embeddings
        except FileNotFoundError:
            logger.info("Creating new embeddings...")
            texts = []
            for _, row in self.catalog.iterrows():
                text = self.create_assessment_text(row)
                texts.append(text)
            
            embeddings = self.model.encode(texts, show_progress_bar=True)
            np.save('data/embeddings.npy', embeddings)
            logger.info(f"Created embeddings for {len(texts)} assessments")
            return embeddings
    
    def create_tfidf_matrix(self):
        """Create TF-IDF matrix for keyword matching"""
        texts = []
        for _, row in self.catalog.iterrows():
            text = f"{row.get('name', '')} {row.get('description', '')}"
            texts.append(text)
        
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def analyze_query(self, query):
        """Analyze query to determine requirements"""
        query_lower = query.lower()
        
        # Technical keywords
        tech_keywords = [
            'java', 'python', 'sql', 'javascript', 'c++', 'coding',
            'programming', 'developer', 'engineer', 'technical',
            'analyst', 'data', 'algorithm', 'database', 'software',
            'frontend', 'backend', 'full stack', 'api', 'cloud'
        ]
        
        # Behavioral keywords
        behav_keywords = [
            'collaborat', 'team', 'communicat', 'leadership', 'manage',
            'stakeholder', 'interpersonal', 'soft skills', 'behavior',
            'personality', 'attitude', 'cultural', 'adaptability',
            'problem-solving', 'critical thinking', 'emotional'
        ]
        
        tech_score = sum(1 for kw in tech_keywords if kw in query_lower)
        behav_score = sum(1 for kw in behav_keywords if kw in query_lower)
        
        # Determine if it's a mixed query
        total = tech_score + behav_score
        is_mixed = tech_score > 0 and behav_score > 0
        
        return {
            'technical_score': tech_score,
            'behavioral_score': behav_score,
            'is_mixed': is_mixed,
            'technical_weight': tech_score / total if total > 0 else 0.5,
            'behavioral_weight': behav_score / total if total > 0 else 0.5
        }
    
    def recommend(self, query, top_k=10):
        """Get recommendations for a query"""
        logger.info(f"Processing query: {query}")
        
        # Analyze query
        query_analysis = self.analyze_query(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate semantic similarity
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Calculate keyword similarity
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Combine scores (70% semantic, 30% keyword)
        combined_scores = 0.7 * semantic_scores + 0.3 * keyword_scores
        
        # Get top candidates
        top_indices = np.argsort(combined_scores)[-20:][::-1]
        
        # Score and balance candidates
        candidates = []
        
        for idx in top_indices:
            assessment = self.catalog.iloc[idx].to_dict()
            base_score = combined_scores[idx]
            
            # Apply boosts based on query analysis
            if query_analysis['is_mixed']:
                test_type = assessment.get('test_type', [])
                if isinstance(test_type, str):
                    test_type = [test_type]
                
                # Boost if it matches the underrepresented type
                if query_analysis['technical_weight'] > 0.6 and 'K' in test_type:
                    base_score *= 1.2
                elif query_analysis['behavioral_weight'] > 0.6 and ('P' in test_type or 'B' in test_type):
                    base_score *= 1.2
            else:
                # Single type query - boost matching assessments
                if query_analysis['technical_score'] > query_analysis['behavioral_score']:
                    if 'K' in str(assessment.get('test_type', '')):
                        base_score *= 1.3
                else:
                    if 'P' in str(assessment.get('test_type', '')) or 'B' in str(assessment.get('test_type', '')):
                        base_score *= 1.3
            
            candidates.append((base_score, idx, assessment))
        
        # Sort by final score
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Format recommendations
        recommendations = []
        seen_urls = set()
        
        for score, idx, assessment in candidates[:top_k]:
            url = assessment.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                # Format test type
                test_type = assessment.get('test_type', [])
                if isinstance(test_type, str):
                    try:
                        test_type = eval(test_type)
                    except:
                        test_type = [test_type]
                
                # Ensure test_type is a list
                if not isinstance(test_type, list):
                    test_type = [test_type] if test_type else []
                
                # Clean up test_type values
                test_type = [str(t).strip() for t in test_type if pd.notna(t)]
                
                recommendations.append({
                    'url': url,
                    'name': str(assessment.get('name', '')),
                    'adaptive_support': str(assessment.get('adaptive_support', 'No')),
                    'description': str(assessment.get('description', ''))[:200] + '...' if len(str(assessment.get('description', ''))) > 200 else str(assessment.get('description', '')),
                    'duration': int(assessment.get('duration', 0)) if pd.notna(assessment.get('duration')) else 0,
                    'remote_support': str(assessment.get('remote_support', 'No')),
                    'test_type': test_type
                })
        
        return recommendations[:10]
    
    def batch_recommend(self, queries):
        """Generate recommendations for multiple queries"""
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            recommendations = self.recommend(query)
            
            for rec in recommendations:
                results.append({
                    'Query': query,
                    'Assessment_url': rec['url']
                })
        
        return pd.DataFrame(results)