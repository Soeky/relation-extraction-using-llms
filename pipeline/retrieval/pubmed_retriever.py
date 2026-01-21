"""PubMed retriever for fetching abstracts from PubMed API."""

import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

try:
    from Bio import Entrez
    from Bio.Entrez import efetch, esearch
except ImportError:
    Entrez = None
    efetch = None
    esearch = None

from .base import Retriever
from config import Config


class PubMedRetriever(Retriever):
    """PubMed API retriever for fetching biomedical abstracts."""
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        max_results: int = 1000,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize PubMed retriever.
        
        Args:
            email: Email for Entrez API (required by NCBI)
            api_key: Optional API key for higher rate limits
            max_results: Maximum number of results to retrieve per query
            logger: Optional logger instance
        """
        if Entrez is None:
            raise ImportError(
                "biopython is required for PubMed retrieval. "
                "Install it with: pip install biopython"
            )
        
        self.email = email or "your.email@example.com"  # NCBI requires email
        self.api_key = api_key
        self.max_results = max_results
        self.logger = logger or logging.getLogger(__name__)
        
        # Set Entrez parameters
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        
        # Rate limiting: NCBI allows 3 requests per second without API key
        # With API key: 10 requests per second
        self.delay = 0.34 if not api_key else 0.1
        self.last_request_time = 0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting for NCBI API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_pubmed(
        self,
        query: str,
        max_results: Optional[int] = None,
        retmax: int = 100,
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results (overrides instance default)
            retmax: Results per batch (max 10000)
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        max_results = max_results or self.max_results
        all_pmids = []
        
        try:
            self._rate_limit()
            self.logger.info(f"Searching PubMed with query: {query}")
            
            # Initial search
            search_handle = esearch(
                db="pubmed",
                term=query,
                retmax=min(retmax, max_results),
                retmode="xml",
                usehistory="y"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            total_found = int(search_results["Count"])
            self.logger.info(f"Found {total_found} results in PubMed")
            
            # Get all PMIDs using history
            if "WebEnv" in search_results and "QueryKey" in search_results:
                webenv = search_results["WebEnv"]
                query_key = search_results["QueryKey"]
                
                # Fetch in batches
                batch_size = min(retmax, max_results)
                for start in range(0, min(total_found, max_results), batch_size):
                    self._rate_limit()
                    fetch_handle = efetch(
                        db="pubmed",
                        rettype="medline",
                        retmode="xml",
                        retstart=start,
                        retmax=batch_size,
                        webenv=webenv,
                        query_key=query_key
                    )
                    batch_results = Entrez.read(fetch_handle)
                    fetch_handle.close()
                    
                    # Extract PMIDs from batch
                    batch_pmids = [
                        str(article["MedlineCitation"]["PMID"])
                        for article in batch_results["PubmedArticle"]
                    ]
                    all_pmids.extend(batch_pmids)
                    
                    self.logger.debug(f"Retrieved {len(batch_pmids)} PMIDs (total: {len(all_pmids)})")
            else:
                # Fallback: use PMIDs directly from search
                all_pmids = search_results["IdList"][:max_results]
            
            self.logger.info(f"Retrieved {len(all_pmids)} PMIDs total")
            return all_pmids[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            raise
    
    def fetch_abstracts(
        self,
        pmids: List[str],
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch abstracts for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            batch_size: Number of abstracts to fetch per batch
            
        Returns:
            List of documents with title, abstract, and metadata
        """
        documents = []
        
        # Fetch in batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                self._rate_limit()
                self.logger.debug(f"Fetching abstracts for batch {i//batch_size + 1} ({len(batch_pmids)} PMIDs)")
                
                fetch_handle = efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="abstract",
                    retmode="xml"
                )
                articles = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                # Parse articles
                for article in articles["PubmedArticle"]:
                    doc = self._parse_article(article)
                    if doc:
                        documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        self.logger.info(f"Fetched {len(documents)} abstracts")
        return documents
    
    def _parse_article(self, article: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a PubMed article into a document dict.
        
        Args:
            article: Parsed XML article from Entrez
            
        Returns:
            Document dict or None if parsing fails
        """
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            
            # Extract title
            title = ""
            if "Article" in medline and "ArticleTitle" in medline["Article"]:
                title = str(medline["Article"]["ArticleTitle"])
            
            # Extract abstract
            abstract = ""
            if "Article" in medline and "Abstract" in medline["Article"]:
                abstract_parts = medline["Article"]["Abstract"].get("AbstractText", [])
                if isinstance(abstract_parts, list):
                    abstract = " ".join(str(part) for part in abstract_parts)
                else:
                    abstract = str(abstract_parts)
            
            # Extract authors
            authors = []
            if "Article" in medline and "AuthorList" in medline["Article"]:
                for author in medline["Article"]["AuthorList"]:
                    if "LastName" in author and "ForeName" in author:
                        authors.append(f"{author['ForeName']} {author['LastName']}")
            
            # Extract journal
            journal = ""
            if "Article" in medline and "Journal" in medline["Article"]:
                journal = medline["Article"]["Journal"].get("Title", "")
            
            # Extract year
            year = ""
            if "Article" in medline and "Journal" in medline["Article"]:
                pub_date = medline["Article"]["Journal"].get("JournalIssue", {}).get("PubDate", {})
                if "Year" in pub_date:
                    year = str(pub_date["Year"])
            
            # Combine title and abstract
            text = f"{title}\n\n{abstract}".strip()
            
            if not text:
                return None
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "text": text,
                "authors": authors,
                "journal": journal,
                "year": year,
                "source": "pubmed",
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing article: {e}")
            return None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve PubMed abstracts matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Search for PMIDs
        pmids = self.search_pubmed(query, max_results=top_k)
        
        if not pmids:
            return []
        
        # Fetch abstracts
        documents = self.fetch_abstracts(pmids)
        
        return documents[:top_k]
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Not applicable for PubMed retriever (documents come from API).
        This method exists for interface compatibility.
        """
        pass
    
    def save_documents_to_files(
        self,
        documents: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """
        Save retrieved documents to files in the RAG sources directory.
        
        Args:
            documents: List of document dicts
            output_dir: Directory to save files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            pmid = doc.get("pmid", "unknown")
            filename = f"{pmid}.txt"
            filepath = output_dir / filename
            
            # Format document text
            content = f"PMID: {pmid}\n"
            if doc.get("title"):
                content += f"Title: {doc['title']}\n"
            if doc.get("journal"):
                content += f"Journal: {doc['journal']}\n"
            if doc.get("year"):
                content += f"Year: {doc['year']}\n"
            if doc.get("authors"):
                content += f"Authors: {', '.join(doc['authors'][:5])}\n"  # First 5 authors
            content += "\n"
            content += doc.get("text", "")
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.logger.debug(f"Saved document to {filepath}")
