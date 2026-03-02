from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import os

from app.recommender import SHLRecommender

# Initialize FastAPI
app = FastAPI(
    title="SHL Assessment Recommendation Engine",
    description="Recommend SHL assessments based on job descriptions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender
recommender = SHLRecommender()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class HealthResponse(BaseModel):
    status: str
    total_assessments: int

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        total_assessments=len(recommender.catalog)
    )

@app.post("/recommend", response_model=List[AssessmentResponse])
async def get_recommendations(request: QueryRequest):
    """Get assessment recommendations"""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        recommendations = recommender.recommend(request.query)
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHL Assessment Recommender</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            
            .input-section {
                margin-bottom: 30px;
            }
            
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                min-height: 120px;
                transition: border-color 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                border-radius: 25px;
                cursor: pointer;
                transition: transform 0.2s, background 0.2s;
                margin-top: 10px;
            }
            
            button:hover {
                background: #5a67d8;
                transform: translateY(-2px);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .loading-spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                color: #e53e3e;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                display: none;
            }
            
            .results {
                margin-top: 30px;
            }
            
            .results h2 {
                color: #333;
                margin-bottom: 20px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            th {
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 500;
            }
            
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
            }
            
            tr:hover {
                background: #f7fafc;
            }
            
            a {
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }
            
            a:hover {
                text-decoration: underline;
            }
            
            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
                margin: 2px;
            }
            
            .badge.K { background: #c6f6d5; color: #22543d; }
            .badge.P { background: #fed7d7; color: #742a2a; }
            .badge.B { background: #feebc8; color: #744210; }
            .badge.S { background: #e9d8fd; color: #44337a; }
            .badge.E { background: #bee3f8; color: #2c5282; }
            
            .examples {
                margin: 20px 0;
                padding: 15px;
                background: #f7fafc;
                border-radius: 10px;
            }
            
            .example-tag {
                display: inline-block;
                padding: 5px 10px;
                background: #e2e8f0;
                border-radius: 15px;
                margin: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }
            
            .example-tag:hover {
                background: #cbd5e0;
            }
            
            .stats {
                display: flex;
                gap: 20px;
                margin-top: 30px;
                padding: 20px;
                background: #f7fafc;
                border-radius: 10px;
            }
            
            .stat-item {
                flex: 1;
                text-align: center;
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            
            .stat-label {
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 SHL Assessment Recommendation Engine</h1>
            <p class="subtitle">Enter a job description or natural language query to get relevant SHL assessments</p>
            
            <div class="examples">
                <strong>Try these examples:</strong>
                <div>
                    <span class="example-tag" onclick="setExample(1)">Java Developer with collaboration skills</span>
                    <span class="example-tag" onclick="setExample(2)">Python, SQL and JavaScript developer</span>
                    <span class="example-tag" onclick="setExample(3)">Cognitive and personality tests for analyst</span>
                </div>
            </div>
            
            <div class="input-section">
                <textarea id="query" placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams."></textarea>
                <button onclick="getRecommendations()" id="submitBtn">Get Recommendations</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <p>Analyzing your query and finding the best assessments...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="results" id="results"></div>
            
            <div class="stats" id="stats" style="display: none;">
                <div class="stat-item">
                    <div class="stat-value" id="totalCount">0</div>
                    <div class="stat-label">Total Assessments</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="technicalCount">0</div>
                    <div class="stat-label">Technical (K)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="behavioralCount">0</div>
                    <div class="stat-label">Behavioral (P/B)</div>
                </div>
            </div>
        </div>
        
        <script>
            function setExample(type) {
                const examples = [
                    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
                    "Looking for mid-level professionals who are proficient in Python, SQL and JavaScript.",
                    "I am hiring for an analyst and want to screen applications using Cognitive and personality tests."
                ];
                document.getElementById('query').value = examples[type - 1];
            }
            
            async function getRecommendations() {
                const query = document.getElementById('query').value.trim();
                if (!query) {
                    alert('Please enter a query');
                    return;
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('error').style.display = 'none';
                document.getElementById('results').innerHTML = '';
                document.getElementById('stats').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;
                
                try {
                    const response = await fetch('/recommend', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Error getting recommendations');
                    }
                    
                    displayResults(data);
                } catch (error) {
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('error').textContent = 'Error: ' + error.message;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                }
            }
            
            function displayResults(recommendations) {
                if (!recommendations || recommendations.length === 0) {
                    document.getElementById('results').innerHTML = '<p>No recommendations found. Try a different query.</p>';
                    return;
                }
                
                let html = '<h2>📋 Recommended Assessments</h2>';
                html += '<table>';
                html += '<tr><th>Assessment</th><th>Duration</th><th>Test Type</th><th>Remote</th><th>Adaptive</th></tr>';
                
                let technicalCount = 0;
                let behavioralCount = 0;
                
                recommendations.forEach(rec => {
                    // Count test types for stats
                    if (rec.test_type.includes('K')) technicalCount++;
                    if (rec.test_type.includes('P') || rec.test_type.includes('B')) behavioralCount++;
                    
                    // Create badge HTML for test types
                    const badges = rec.test_type.map(type => 
                        `<span class="badge ${type}">${type}</span>`
                    ).join('');
                    
                    html += '<tr>';
                    html += `<td><a href="${rec.url}" target="_blank">${rec.name}</a></td>`;
                    html += `<td>${rec.duration} min</td>`;
                    html += `<td>${badges}</td>`;
                    html += `<td>${rec.remote_support}</td>`;
                    html += `<td>${rec.adaptive_support}</td>`;
                    html += '</tr>';
                });
                
                html += '</table>';
                document.getElementById('results').innerHTML = html;
                
                // Update stats
                document.getElementById('totalCount').textContent = recommendations.length;
                document.getElementById('technicalCount').textContent = technicalCount;
                document.getElementById('behavioralCount').textContent = behavioralCount;
                document.getElementById('stats').style.display = 'flex';
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)