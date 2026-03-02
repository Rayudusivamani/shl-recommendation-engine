import pytest
from app.retriever.embedding_retriever import SHLRetriever

def test_retriever_initialization():
    retriever = SHLRetriever()
    assert retriever.model is not None

def test_hybrid_retrieval():
    retriever = SHLRetriever()
    # Add test logic here
    pass