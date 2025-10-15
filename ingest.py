import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
BASE_URL = "https://en.wikipedia.org"
MAX_PAGES = 100         # Maximum pages to crawl
MAX_WORKERS = 5         # Number of threads for parallel fetching
REQUEST_DELAY = 1       # Seconds between requests
USER_AGENT = "MyLangChainCrawler/1.0 (+https://example.com)"

HEADERS = {"User-Agent": USER_AGENT}

def crawl_website(base_url, max_pages=50):
    """Crawl a website politely with throttling."""
    visited = set()
    to_visit = [base_url]

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=5)
            if response.status_code != 200:
                continue
            visited.add(url)
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(base_url, href)
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)
            time.sleep(REQUEST_DELAY)  # Throttle requests
        except requests.exceptions.RequestException:
            continue
    return list(visited)

def load_page(url):
    """Load a single page using LangChain WebBaseLoader."""
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

# Crawl the website
print(f"Crawling {BASE_URL}...")
all_pages = crawl_website(BASE_URL, MAX_PAGES)
print(f"Found {len(all_pages)} pages.")

# Load pages in parallel (still throttled per crawl)
documents = []
print("Loading pages...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(load_page, url) for url in all_pages]
    for future in as_completed(futures):
        documents.extend(future.result())

print(f"Loaded {len(documents)} documents.")

# Split documents into chunks
text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"Split documents into {len(texts)} text chunks.")

# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")

print("Vector store created and saved to faiss_index.")
