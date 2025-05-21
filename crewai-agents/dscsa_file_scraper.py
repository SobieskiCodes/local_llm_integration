"""
Enhanced DSCSA Web Scraper with file download capabilities

This module extends the basic scraper with the ability to:
1. Identify links to documents (PDFs, DOCs, etc.) within web pages
2. Download these documents
3. Extract text from these documents
4. Store both the document content and metadata
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import re
import os
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import mimetypes
from pathlib import Path

# For text extraction from various file types
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: unstructured library not available. Text extraction from PDFs and docs will be limited.")
    
# File types to look for and download
DOCUMENT_EXTENSIONS = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.csv']

def setup_selenium_driver():
    """Set up and return a selenium webdriver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def extract_links_from_page(url: str) -> List[Dict[str, str]]:
    """
    Extract all links from a page, with a focus on document links.
    
    Args:
        url: The page URL to scrape for links
        
    Returns:
        A list of dictionaries with link info (url, text, type)
    """
    driver = setup_selenium_driver()
    links = []
    
    try:
        driver.get(url)
        time.sleep(3)  # Wait for JS to load
        
        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            link_text = a_tag.get_text(strip=True)
            
            # Skip empty or javascript links
            if not href or href.startswith('javascript:') or href == '#':
                continue
                
            # Make absolute URL if needed
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(url, href)
            
            # Determine if it's a document link
            is_document = False
            file_extension = None
            
            # Check by extension
            for ext in DOCUMENT_EXTENSIONS:
                if href.lower().endswith(ext):
                    is_document = True
                    file_extension = ext
                    break
            
            # Check by text if no extension match
            if not is_document:
                doc_indicators = ['pdf', 'document', 'guidance', 'form', 'publication', 
                                 'worksheet', 'template', 'download']
                if any(indicator in link_text.lower() for indicator in doc_indicators):
                    is_document = True
            
            links.append({
                'url': href,
                'text': link_text,
                'is_document': is_document,
                'extension': file_extension
            })
            
        return links
    
    finally:
        driver.quit()

def download_file(url: str, output_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Download a file from a URL.
    
    Args:
        url: The URL of the file to download
        output_dir: Directory to save the file (uses temp dir if None)
        
    Returns:
        Tuple of (file_path, error_message)
    """
    try:
        # Use a temporary directory if none specified
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        # Make sure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename from URL or generate one
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            # Generate a filename from the URL
            filename = f"document_{hash(url) % 10000}.bin"
        
        file_path = os.path.join(output_dir, filename)
        
        # Download the file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Determine file extension if not clear from URL
        if '.' not in filename:
            content_type = response.headers.get('content-type')
            if content_type:
                ext = mimetypes.guess_extension(content_type.split(';')[0].strip())
                if ext:
                    filename += ext
                    file_path = os.path.join(output_dir, filename)
        
        # Save the file
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return file_path, None
    
    except Exception as e:
        return None, f"Error downloading {url}: {str(e)}"

def extract_text_from_file(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text from a downloaded file.
    
    Args:
        file_path: Path to the downloaded file
        
    Returns:
        Tuple of (extracted_text, error_message)
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        # Use unstructured library if available
        if UNSTRUCTURED_AVAILABLE:
            elements = partition(file_path)
            return "\n".join([str(el) for el in elements]), None
            
        # Simple fallback for text files
        elif file_extension in ['.txt', '.csv']:
            with open(file_path, 'r', errors='ignore') as f:
                return f.read(), None
        
        # For other file types without unstructured
        else:
            return None, f"Cannot extract text from {file_extension} files without unstructured library"
    
    except Exception as e:
        return None, f"Error extracting text from {file_path}: {str(e)}"

def scrape_page_with_files(url: str, download_docs: bool = True, 
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive page scraper that gets page text and linked documents.
    
    Args:
        url: The website URL to scrape
        download_docs: Whether to download and extract text from linked documents
        output_dir: Directory to save downloaded files
        
    Returns:
        Dictionary with page text and document information
    """
    # Initialize result structure
    result = {
        'url': url,
        'page_text': '',
        'documents': [],
        'errors': []
    }
    
    try:
        # First scrape the main page text using Selenium
        driver = setup_selenium_driver()
        driver.get(url)
        time.sleep(3)  # Wait for JS to load
        
        # Get the page text
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text
        result['page_text'] = soup.get_text(separator='\n', strip=True)
        
        # Extract links
        links = extract_links_from_page(url)
        
        # Process document links if requested
        if download_docs:
            for link in links:
                if link['is_document']:
                    doc_result = {
                        'url': link['url'],
                        'title': link['text'],
                        'extension': link['extension'],
                        'downloaded': False,
                        'text_extracted': False,
                        'text': None,
                        'local_path': None,
                        'error': None
                    }
                    
                    # Download the file
                    file_path, download_error = download_file(link['url'], output_dir)
                    
                    if download_error:
                        doc_result['error'] = download_error
                        result['errors'].append(download_error)
                    else:
                        doc_result['downloaded'] = True
                        doc_result['local_path'] = file_path
                        
                        # Extract text if downloaded successfully
                        text, extract_error = extract_text_from_file(file_path)
                        
                        if extract_error:
                            doc_result['error'] = extract_error
                            result['errors'].append(extract_error)
                        else:
                            doc_result['text_extracted'] = True
                            doc_result['text'] = text
                    
                    result['documents'].append(doc_result)
        
        return result
        
    except Exception as e:
        error_msg = f"Error scraping {url}: {str(e)}"
        result['errors'].append(error_msg)
        return result
    finally:
        if 'driver' in locals():
            driver.quit()

def scrape_dscsa_sites_with_files(urls: List[str], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scrape multiple DSCSA websites including their linked documents.
    
    Args:
        urls: List of URLs to scrape
        output_dir: Directory to save downloaded files
        
    Returns:
        List of results for each URL
    """
    results = []
    
    for url in urls:
        print(f"Scraping {url}...")
        result = scrape_page_with_files(url, download_docs=True, output_dir=output_dir)
        results.append(result)
        
    return results

def prepare_documents_for_storage(scrape_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare scraped data for storage in vector database.
    
    Args:
        scrape_results: Results from scrape_dscsa_sites_with_files
        
    Returns:
        List of documents ready for storage
    """
    storage_docs = []
    
    for site_result in scrape_results:
        # Store the main page text
        main_doc = {
            'source': site_result['url'],
            'type': 'dscsa_web_page',
            'text': site_result['page_text'],
            'metadata': {
                'url': site_result['url'],
                'document_count': len(site_result['documents']),
                'has_errors': len(site_result['errors']) > 0
            }
        }
        storage_docs.append(main_doc)
        
        # Store each document text as a separate document
        for doc in site_result['documents']:
            if doc['text_extracted'] and doc['text']:
                doc_storage = {
                    'source': doc['url'],
                    'type': 'dscsa_document',
                    'text': doc['text'],
                    'metadata': {
                        'title': doc['title'],
                        'url': doc['url'], 
                        'parent_page': site_result['url'],
                        'extension': doc['extension']
                    }
                }
                storage_docs.append(doc_storage)
    
    return storage_docs

# Example usage
if __name__ == "__main__":
    # Define test URLs
    test_urls = [
        "https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa/drug-supply-chain-security-act-law-and-policies",
        "https://ref.gs1.org/standards/epcis/",
        "https://ref.gs1.org/standards/cbv/"
    ]
    
    # Create an output directory
    output_dir = os.path.join(os.getcwd(), "downloaded_dscsa_docs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Scrape the sites
    results = scrape_dscsa_sites_with_files(test_urls, output_dir)
    
    # Prepare for storage
    storage_docs = prepare_documents_for_storage(results)
    
    # Print some stats
    print(f"Scraped {len(results)} sites")
    total_docs = sum(len(r['documents']) for r in results)
    print(f"Found {total_docs} linked documents")
    successful_docs = sum(1 for r in results for d in r['documents'] if d['text_extracted'])
    print(f"Successfully extracted text from {successful_docs} documents")
    
    # Print document titles
    print("\nDocuments found:")
    for r in results:
        for d in r['documents']:
            status = "✅" if d['text_extracted'] else "❌"
            print(f"{status} {d['title']} ({d['url']})")