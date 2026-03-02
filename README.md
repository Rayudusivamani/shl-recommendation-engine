# SHL Assessment Recommendation Engine

An intelligent recommendation system that maps natural language queries and job descriptions to relevant SHL individual assessments.

## Features

- Web scraping of SHL product catalog (400+ individual assessments)
- Semantic search using sentence-transformers embeddings
- Intent detection for technical vs behavioral skills
- Balanced recommendations for mixed queries
- REST API with FastAPI
- Simple web interface for testing
- Mean Recall@10 evaluation on provided train set

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shl-recommendation-engine.git
cd shl-recommendation-engine