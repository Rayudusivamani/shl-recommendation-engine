import time
import json
import logging
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLCrawler:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.assessments = []
        self.scraped_urls = set() # To prevent duplicates

    def crawl_catalog(self):
        logger.info("Launching headless browser...")
        chrome_options = Options()
        # chrome_options.add_argument("--headless") # Commented out so you can watch it work! 
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        try:
            logger.info(f"Loading {self.catalog_url}...")
            driver.get(self.catalog_url)
            time.sleep(5) # Wait for initial load
            
            page_number = 1
            while True:
                logger.info(f"Scraping Page {page_number}...")
                
                # 1. Parse the current page HTML
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # 2. Find all table rows based on your screenshot!
                rows = soup.find_all('tr')
                page_items_found = 0
                
                for row in rows:
                    # Look for the specific <td> that holds the title
                    title_td = row.find('td', class_='custom__table-heading__title')
                    if not title_td:
                        continue # Skip rows that aren't assessments
                        
                    link_tag = title_td.find('a')
                    if not link_tag:
                        continue
                        
                    url = urljoin(self.base_url, link_tag.get('href', ''))
                    name = link_tag.text.strip()
                    
                    # Filter out the Pre-packaged solutions (The assignment requires this!)
                    if "Solution" in name or "Pre-packaged" in name:
                        continue

                    if url not in self.scraped_urls:
                        self.scraped_urls.add(url)
                        page_items_found += 1
                        
                        # Add to our final list
                        self.assessments.append({
                            "url": url,
                            "name": name,
                            "adaptive_support": "No",       # Defaulting for now
                            "description": f"{name} assessment", # Placeholder
                            "duration": 30,                 # Placeholder
                            "remote_support": "Yes",        # Defaulting for now
                            "test_type": ["Knowledge & Skills"] # Placeholder
                        })

                logger.info(f"Found {page_items_found} individual assessments on Page {page_number}.")
                
                # 3. Find and click the "Next" button to go to the next page
                try:
                    # Look for the Next button using a broader XPath
                    next_button = driver.find_element(By.XPATH, "//*[contains(text(), 'Next') or contains(@class, 'next')]")
                    
                    # BROADENED CHECK: Stop if button is disabled, has 'disabled' class, or if we passed Page 12
                    button_class = next_button.get_attribute("class") or ""
                    is_disabled = next_button.get_attribute("disabled") is not None
                    
                    if "disabled" in button_class.lower() or is_disabled or page_number >= 12:
                        logger.info("Reached the last page! Stopping pagination.")
                        break
                        
                    # Use a JavaScript click to bypass any overlays
                    driver.execute_script("arguments[0].click();", next_button)
                    
                    logger.info(f"Clicked 'Next'. Waiting for Page {page_number + 1} to load...")
                    time.sleep(5) # Give it 5 seconds to load the new table content
                    page_number += 1
                    
                except Exception as e:
                    logger.info("No more 'Next' pages found. Finishing scrape.")
                    break

            logger.info(f"SUCCESS! Scraped a total of {len(self.assessments)} individual assessments.")
            
        except Exception as e:
            logger.error(f"Crawler failed: {e}")
        finally:
            driver.quit() # Always close the browser
            
        return self.assessments

    def save_to_file(self, filename="shl_catalog.json"):
        import os
        os.makedirs("data/raw", exist_ok=True)
        filepath = f"data/raw/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved data to {filepath}")

if __name__ == "__main__":
    crawler = SHLCrawler()
    crawler.crawl_catalog()
    crawler.save_to_file()