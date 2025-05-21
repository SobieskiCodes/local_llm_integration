"""
DSCSA Web Scraper Utility

This module provides functionality for scraping DSCSA-related information from 
various websites and preparing it for storage in a knowledge base.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from typing import List, Dict, Any, Optional

# List of common DSCSA-related websites
DSCSA_SOURCES = [
    # DSCSA compliance sources
    "https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa",
    "https://www.hda.org/pharmaceutical-traceability-hub",
    "https://www.gs1us.org/industries-and-insights/by-industry/healthcare/standards-in-use/pharmaceutical",
    "https://www.cdc.gov/vaccines/programs/vtrcks/awardees/dscsa.html",
    "https://www.pharmacist.com/Practice/DSCSA",
    "https://www.lsps.com/blog/dscsa-pharmaceutical-serialization-requirements-2023/",
    
    # EPCIS-specific sources
    "https://www.gs1.org/standards/epcis",
    "https://www.gs1us.org/industries-and-insights/by-industry/healthcare/standards-in-use/epcis",
    "https://www.tracelink.com/solutions/healthcare/epcis-data-exchange",
    "https://www.c4scs.org/epcis-for-dscsa-with-help-videos/",
    "https://www.rxtrace.com/category/epcis/",
    "https://www.lsps.com/services-pharmaceutical-serialization/epcis/"
]

def setup_selenium_driver():
    """Set up and return a selenium webdriver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scrape_with_selenium(url: str) -> str:
    """
    Scrape a URL using Selenium for better handling of JavaScript-rendered content.
    
    Args:
        url: The website URL to scrape
        
    Returns:
        The extracted text content from the page
    """
    driver = setup_selenium_driver()
    try:
        driver.get(url)
        # Wait for dynamic content to load
        time.sleep(3)
        page_source = driver.page_source
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        return text
    finally:
        driver.quit()

def scrape_with_requests(url: str) -> str:
    """
    Scrape a URL using Requests and BeautifulSoup.
    
    Args:
        url: The website URL to scrape
        
    Returns:
        The extracted text content from the page
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error scraping {url}: {str(e)}"

def extract_dscsa_info(text: str) -> Dict[str, List[str]]:
    """
    Extract DSCSA-specific information from scraped text.
    
    Args:
        text: The raw text content from a scraped page
        
    Returns:
        A dictionary of categorized DSCSA information
    """
    # Define DSCSA-related keywords and categories
    categories = {
        "requirements": [
            "requirement", "compliance", "must", "shall", "required", "provision", 
            "mandate", "obligation", "compulsory", "necessary", "essential"
        ],
        "deadlines": [
            "deadline", "date", "timeline", "schedule", "by", "due", "until", 
            "implementation date", "effective date", "compliance date"
        ],
        "stakeholders": [
            "manufacturer", "wholesaler", "distributor", "pharmacy", "dispenser", 
            "repackager", "provider", "3PL", "third-party", "logistics"
        ],
        "identifiers": [
            "identifier", "SNI", "serialization", "serial", "GTIN", "NDC", "barcode", 
            "DataMatrix", "2D", "product identifier", "unique"
        ],
        "transactions": [
            "transaction", "T3", "TI", "transaction information", "transaction statement", 
            "transaction history", "EPCIS", "verification", "trace", "tracking"
        ]
    }
    
    # Initialize results dictionary
    results = {category: [] for category in categories}
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Categorize sentences
    for sentence in sentences:
        for category, keywords in categories.items():
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                # Don't add duplicate sentences
                if sentence not in results[category]:
                    results[category].append(sentence)
    
    return results

def create_structured_document(url: str, extracted_info: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create a structured document from extracted information for storage.
    
    Args:
        url: The source URL
        extracted_info: The categorized DSCSA information
        
    Returns:
        A structured document ready for storage
    """
    # Create the document
    document = {
        "source": url,
        "type": "dscsa_scraped",
        "categories": {}
    }
    
    # Add the categorized content
    for category, sentences in extracted_info.items():
        document["categories"][category] = sentences
        
    return document

def generate_documents_from_text(url: str, text: str) -> List[Dict[str, Any]]:
    """
    Generate smaller documents from a large text for better processing.
    
    Args:
        url: The source URL
        text: The full text content
        
    Returns:
        A list of document chunks
    """
    # Extract categorized information
    categorized_info = extract_dscsa_info(text)
    
    # Create the main document
    main_doc = create_structured_document(url, categorized_info)
    
    # Create smaller chunks if needed (for large texts)
    docs = [main_doc]
    
    # Flatten all sentences
    all_sentences = []
    for sentences in categorized_info.values():
        all_sentences.extend(sentences)
    
    # If we have a lot of sentences, create additional chunks
    if len(all_sentences) > 20:
        # Create chunks of 20 sentences
        for i in range(0, len(all_sentences), 20):
            chunk = all_sentences[i:i+20]
            docs.append({
                "source": url,
                "type": "dscsa_chunk",
                "text": " ".join(chunk)
            })
    
    return docs

def scrape_dscsa_sites(urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Scrape multiple DSCSA-related websites and prepare documents for storage.
    
    Args:
        urls: List of URLs to scrape (uses defaults if None)
        
    Returns:
        A list of documents ready for storage
    """
    if urls is None:
        urls = DSCSA_SOURCES
        
    all_documents = []
    
    for url in urls:
        try:
            # Try with Selenium first
            try:
                text = scrape_with_selenium(url)
            except Exception:
                # Fall back to requests if Selenium fails
                text = scrape_with_requests(url)
                
            # Generate documents from the text
            documents = generate_documents_from_text(url, text)
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
    return all_documents

if __name__ == "__main__":
    # For testing
    documents = scrape_dscsa_sites()
    print(f"Generated {len(documents)} documents from DSCSA sources")