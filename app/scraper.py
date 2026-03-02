import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import List, Dict
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = f"{self.base_url}/solutions/products/product-catalog/"
        self.assessments = []
        
    def parse_duration(self, text):
        """Extract duration from text"""
        if not text:
            return 0
        
        # Look for patterns like "30 minutes", "45 min", etc.
        patterns = [
            r'(\d+)\s*minutes?',
            r'(\d+)\s*min',
            r'(\d+)(?=\s*min)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0
    
    def parse_test_type(self, text):
        """Extract test type codes"""
        if not text:
            return ['G']  # General
        
        text_lower = text.lower()
        types = []
        
        # Mapping of keywords to test type codes
        type_mapping = {
            'knowledge': 'K',
            'skills': 'K',
            'cognitive': 'K',
            'ability': 'K',
            'aptitude': 'K',
            'technical': 'K',
            'personality': 'P',
            'behavioral': 'B',
            'behaviour': 'B',
            'simulation': 'S',
            'exercise': 'E',
            'situational': 'S'
        }
        
        for keyword, code in type_mapping.items():
            if keyword in text_lower:
                if code not in types:
                    types.append(code)
        
        return types if types else ['G']
    
    def scrape_catalog(self):
        """Main scraping function"""
        logger.info(f"Starting to scrape {self.catalog_url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.catalog_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all product cards/rows (adjust selector based on actual structure)
            # You'll need to inspect the website to find the correct selectors
            product_cards = soup.find_all('div', class_='product-card')  # Adjust this selector
            
            for card in product_cards:
                try:
                    # Check if it's individual test (not pre-packaged)
                    if 'pre-packaged' in card.get_text().lower():
                        continue
                    
                    # Extract name and URL
                    name_elem = card.find('a', class_='product-title')
                    if not name_elem:
                        continue
                    
                    name = name_elem.get_text().strip()
                    relative_url = name_elem.get('href', '')
                    full_url = urljoin(self.base_url, relative_url)
                    
                    # Extract basic info from card
                    description_elem = card.find('p', class_='description')
                    description = description_elem.get_text().strip() if description_elem else ''
                    
                    # Get additional details
                    details = self.scrape_assessment_details(full_url)
                    
                    # Create assessment record
                    assessment = {
                        'name': name,
                        'url': full_url,
                        'description': description or details.get('description', ''),
                        'duration': details.get('duration', self.parse_duration(card.get_text())),
                        'adaptive_support': details.get('adaptive', 'No'),
                        'remote_support': details.get('remote', 'No'),
                        'test_type': details.get('test_type', self.parse_test_type(description)),
                    }
                    
                    self.assessments.append(assessment)
                    logger.info(f"Scraped: {name}")
                    
                    # Be respectful - don't hammer the server
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error scraping card: {e}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(self.assessments)
            logger.info(f"Scraped {len(df)} assessments")
            
            # Save to CSV
            df.to_csv('data/shl_catalog.csv', index=False)
            return df
            
        except Exception as e:
            logger.error(f"Error scraping catalog: {e}")
            return pd.DataFrame()
    
    def scrape_assessment_details(self, url):
        """Scrape individual assessment details"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find description
            desc_elem = soup.find('meta', {'name': 'description'})
            description = desc_elem.get('content', '') if desc_elem else ''
            
            if not description:
                desc_elem = soup.find('div', class_='description')
                description = desc_elem.get_text().strip() if desc_elem else ''
            
            # Find duration
            duration = 0
            duration_patterns = soup.find_all(text=re.compile(r'\d+\s*minutes?', re.I))
            if duration_patterns:
                duration = self.parse_duration(duration_patterns[0])
            
            # Check for adaptive support
            adaptive = 'Yes' if soup.find(text=re.compile('adaptive', re.I)) else 'No'
            
            # Check for remote support
            remote = 'Yes' if soup.find(text=re.compile('remote|online', re.I)) else 'No'
            
            # Find test type
            test_type_text = ''
            type_elem = soup.find('div', class_='test-type')
            if type_elem:
                test_type_text = type_elem.get_text()
            else:
                # Look in description
                test_type_text = description
            
            test_type = self.parse_test_type(test_type_text)
            
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

# Run scraper
if __name__ == "__main__":
    scraper = SHLScraper()
    df = scraper.scrape_catalog()
    print(f"Successfully scraped {len(df)} assessments")
    print(df.head())
    print(f"\nData saved to data/shl_catalog.csv")