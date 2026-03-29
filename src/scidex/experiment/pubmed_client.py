"""Query PubMed for papers relevant to experiment design.

Uses NCBI E-utilities (free, no key needed for low-volume). Rate limit: 3 req/sec without API key.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)


class PubMedClient:
    """Query PubMed for papers relevant to experiment design."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, max_results: int = 10) -> list[str]:
        """Search PubMed and return list of PMIDs.

        Args:
            query: PubMed search query.
            max_results: Max PMIDs to return.

        Returns:
            List of PMID strings.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 100),
            "retmode": "json",
        }
        try:
            resp = requests.get(self.ESEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("PubMed search failed: %s", e)
            return []

        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """Fetch paper metadata + abstracts for given PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of dicts with keys: pmid, title, abstract, authors, journal, year.
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        try:
            resp = requests.get(self.EFETCH_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("PubMed fetch failed: %s", e)
            return []

        return self._parse_xml(resp.text)

    def search_and_fetch(self, query: str, max_results: int = 5) -> list[dict]:
        """Convenience: search + fetch in one call.

        Args:
            query: PubMed search query.
            max_results: Max papers to return.

        Returns:
            List of paper dicts with metadata and abstracts.
        """
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return []
        return self.fetch_abstracts(pmids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_text: str) -> list[dict]:
        """Parse PubMed efetch XML into a list of paper dicts."""
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.warning("Failed to parse PubMed XML response")
            return []

        for article_el in root.findall(".//PubmedArticle"):
            medline = article_el.find("MedlineCitation")
            if medline is None:
                continue

            article = medline.find("Article")
            if article is None:
                continue

            # PMID
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            # Title
            title_el = article.find("ArticleTitle")
            title = title_el.text if title_el is not None else ""

            # Abstract
            abstract_el = article.find("Abstract/AbstractText")
            abstract = abstract_el.text if abstract_el is not None else ""

            # Authors
            authors = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    if last is not None and fore is not None:
                        authors.append(f"{fore.text} {last.text}")
                    elif last is not None:
                        authors.append(last.text)

            # Journal
            journal_el = article.find("Journal/Title")
            journal = journal_el.text if journal_el is not None else ""

            # Year
            year_el = article.find("Journal/JournalIssue/PubDate/Year")
            year = year_el.text if year_el is not None else ""

            results.append(
                {
                    "pmid": pmid,
                    "title": title or "",
                    "abstract": abstract or "",
                    "authors": authors,
                    "journal": journal or "",
                    "year": year,
                }
            )

        return results
