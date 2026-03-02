# scripts/generate_embeddings.py
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever.embedding_retriever import SHLRetriever
from app.crawler.shl_crawler import SHLCrawler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Check if we need to crawl or use existing data
    data_path = "data/raw/shl_catalog.json"
    
    if not os.path.exists(data_path):
        logger.info("Crawling SHL catalog...")
        crawler = SHLCrawler()
        assessments = crawler.crawl_catalog()
        crawler.save_to_file("shl_catalog.json")
    else:
        logger.info(f"Loading existing data from {data_path}")
        with open(data_path, 'r') as f:
            assessments = json.load(f)
    
    logger.info(f"Loaded {len(assessments)} assessments")
    if len(assessments) == 0:
        logger.error("No assessments found! Stopping execution. Please fix the crawler.")
        return
    
    # Build retriever index
    retriever = SHLRetriever()
    retriever.build_index(assessments)
    
    # Save index
    output_path = "data/processed/index"
    retriever.save(output_path)
    
    logger.info(f"Index saved to {output_path}")
    logger.info(f"Number of vectors: {retriever.index.ntotal}")

if __name__ == "__main__":
    main()