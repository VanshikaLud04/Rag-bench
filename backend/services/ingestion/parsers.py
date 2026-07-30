from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pypdf
import docx
import requests
from bs4 import BeautifulSoup
import uuid
import re
import os
from pathlib import Path

class RawDocument:
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata

class DocumentParser(ABC):
    @abstractmethod
    def parse(self, source: str) -> List[RawDocument]:
        pass

class PDFParser(DocumentParser):
    def parse(self, source: str) -> List[RawDocument]:
        reader = pypdf.PdfReader(source)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = {"source": Path(source).name, "type": "pdf"}
        return [RawDocument(text=text, metadata=metadata)]

class DOCXParser(DocumentParser):
    def parse(self, source: str) -> List[RawDocument]:
        doc = docx.Document(source)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        metadata = {"source": Path(source).name, "type": "docx"}
        return [RawDocument(text=text, metadata=metadata)]

class MarkdownParser(DocumentParser):
    def parse(self, source: str) -> List[RawDocument]:
        with open(source, 'r', encoding='utf-8') as f:
            text = f.read()
        metadata = {"source": Path(source).name, "type": "markdown"}
        return [RawDocument(text=text, metadata=metadata)]

class WebsiteParser(DocumentParser):
    def _can_fetch(self, url: str) -> bool:
        # A basic robots.txt check could go here, for simplicity assuming true if not explicitly forbidden
        from urllib.parse import urlparse
        import urllib.robotparser
        
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except Exception:
            return True # If robots.txt can't be fetched, assume allowed or handle differently

    def parse(self, source: str) -> List[RawDocument]:
        if not self._can_fetch(source):
            raise PermissionError(f"Robots.txt forbids scraping {source}")
            
        response = requests.get(source, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator='\n')
        # Cleanup extra newlines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        metadata = {"source": source, "type": "website", "title": soup.title.string if soup.title else ""}
        return [RawDocument(text=text, metadata=metadata)]

class GithubRepoParser(DocumentParser):
    def parse(self, source: str) -> List[RawDocument]:
        # Simplistic github repo parser that reads local files.
        # Assuming `source` is a local directory path to a cloned repo.
        docs = []
        for root, _, files in os.walk(source):
            if '.git' in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(('.py', '.js', '.ts', '.md', '.txt')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        metadata = {"source": os.path.relpath(file_path, source), "type": "code"}
                        docs.append(RawDocument(text=text, metadata=metadata))
                    except Exception:
                        pass
        return docs

class ParserRegistry:
    def __init__(self):
        self._parsers = {
            "pdf": PDFParser(),
            "docx": DOCXParser(),
            "md": MarkdownParser(),
            "http": WebsiteParser(),
            "https": WebsiteParser(),
            "dir": GithubRepoParser(), # Local directory placeholder
        }

    def get_parser(self, source: str, source_type: Optional[str] = None) -> DocumentParser:
        if source_type:
            return self._parsers.get(source_type.lower())
        
        if source.startswith('http://') or source.startswith('https://'):
            return self._parsers["https"]
        
        if os.path.isdir(source):
            return self._parsers["dir"]
            
        ext = source.split('.')[-1].lower()
        if ext in self._parsers:
            return self._parsers[ext]
            
        raise ValueError(f"No parser available for source: {source}")
