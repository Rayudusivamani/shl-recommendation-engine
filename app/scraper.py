import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = f"{self.base_url}/solutions/products/product-catalog/"
        self.assessments = []
        
    def parse_duration(self, duration_text: str) -> int:
        """Extract duration in minutes from text"""
        if not duration_text:
            return 0
        
        # Find numbers in the text
        numbers = re.findall(r'\d+', duration_text)
        if numbers:
            return int(numbers[0])
        
        # Handle special cases
        if 'minutes' in duration_text.lower():
            return 30  # Default if minutes mentioned but no number
        return 0
    
    def parse_test_type(self, type_text: str) -> List[str]:
        """Map test type descriptions to codes"""
        type_mapping = {
            'knowledge': 'K',
            'skills': 'K',
            'cognitive': 'K',
            'ability': 'K',
            'personality': 'P',
            'behavior': 'B',
            'behaviour': 'B',
            'simulation': 'S',
            'exercise': 'E'
        }
        
        codes = set()
        text_lower = type_text.lower()
        
        for key, code in type_mapping.items():
            if key in text_lower:
                codes.add(code)
        
        return list(codes) if codes else ['G']  # G for General
    
    def scrape_catalog(self) -> pd.DataFrame:
        """Main scraping function"""
        logger.info(f"Starting to scrape {self.catalog_url}")
        
        try:
            response = requests.get(self.catalog_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all assessment rows - this needs adjustment based on actual structure
            assessment_rows = soup.find_all('tr', class_='product-row')
            
            for row in assessment_rows:
                try:
                    # Skip if it's a pre-packaged solution
                    if 'pre-packaged' in row.get_text().lower():
                        continue
                    
                    # Extract basic info
                    name_cell = row.find('td', class_='product-name')
                    if not name_cell:
                        continue
                    
                    name = name_cell.get_text().strip()
                    relative_url = name_cell.find('a')['href'] if name_cell.find('a') else ''
                    full_url = f"{self.base_url}{relative_url}" if relative_url else ''
                    
                    # Get additional details
                    details = self.scrape_assessment_details(full_url)
                    
                    assessment = {
                        'name': name,
                        'url': full_url,
                        'description': details.get('description', ''),
                        'duration': details.get('duration', 0),
                        'adaptive_support': details.get('adaptive', 'No'),
                        'remote_support': details.get('remote', 'No'),
                        'test_type': details.get('test_type', []),
                    }
                    
                    self.assessments.append(assessment)
                    logger.info(f"Scraped: {name}")
                    time.sleep(1)  # Be respectful to the server
                    
                except Exception as e:
                    logger.error(f"Error scraping row: {e}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(self.assessments)
            logger.info(f"Scraped {len(df)} assessments")
            
            # Save to CSV
            df.to_csv('data/catalog.csv', index=False)
            return df
            
        except Exception as e:
            logger.error(f"Error scraping catalog: {e}")
            return pd.DataFrame()
    
    def scrape_assessment_details(self, url: str) -> Dict:
        """Scrape individual assessment details"""
        if not url:
            return {}
        
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract description
            desc_elem = soup.find('meta', {'name': 'description'})
            description = desc_elem['content'] if desc_elem else ''
            
            # Find duration
            duration_elem = soup.find(text=re.compile(r'\d+\s*minutes', re.I))
            duration = self.parse_duration(duration_elem) if duration_elem else 0
            
            # Check adaptive/remote support
            adaptive = 'Yes' if soup.find(text=re.compile('adaptive', re.I)) else 'No'
            remote = 'Yes' if soup.find(text=re.compile('remote', re.I)) else 'No'
            
            # Find test type
            type_elem = soup.find('div', class_='test-type')
            test_type = self.parse_test_type(type_elem.get_text() if type_elem else '')
            
            return {
                'description': description,
                'duration': duration,
                'adaptive': adaptive,
                'remote': remote,
                'test_type': test_type
            }
            
        except Exception as e:
            logger.error(f"Error scraping details from {url}: {e}")
            return {}

# For testing
if __name__ == "__main__":
    scraper = SHLScraper()
    df = scraper.scrape_catalog()
    print(f"Scraped {len(df)} assessments")
    print(df.head())