# scripts/generate_submission.py
import pandas as pd
from app.retriever.embedding_retriever import SHLRetriever

# Load retriever
retriever = SHLRetriever()
retriever.load("data/processed/index")  # ← Loads the pre-built index

# Load test queries
test_df = pd.read_csv("data/evaluation/test.csv")  # ← Reads test queries

# Generate predictions
predictions = []
for query in test_df['query']:  # ← Loops through each test query
    results = retriever.hybrid_retrieve(query, k=10)  # ← Gets recommendations
    for assessment, score in results:
        predictions.append({
            'query': query,
            'Assessment_url': assessment['url']
        })

# Save in required format
pred_df = pd.DataFrame(predictions)
pred_df.to_csv('submission.csv', index=False)  # ← Creates submission file
print("submission.csv created successfully!")