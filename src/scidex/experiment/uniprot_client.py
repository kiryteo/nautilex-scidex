"""Query UniProt for protein information relevant to hypotheses.

Free API, no key needed. Rate limit: ~100 req/sec.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class UniProtClient:
    """Query UniProt for protein information relevant to hypotheses."""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search UniProt for proteins matching a query.

        Args:
            query: Search query (gene name, protein name, keyword).
            limit: Max results to return.

        Returns:
            List of dicts with keys: accession, name, gene_name, organism,
            function, sequence_length.
        """
        params = {
            "query": query,
            "format": "json",
            "size": min(limit, 25),
        }
        try:
            resp = requests.get(f"{self.BASE_URL}/search", params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("UniProt search failed: %s", e)
            return []

        data = resp.json()
        results = []
        for entry in data.get("results", []):
            results.append(self._parse_entry(entry))
        return results

    def get_protein(self, accession: str) -> dict | None:
        """Get detailed protein info by UniProt accession.

        Args:
            accession: UniProt accession ID (e.g., "P04637").

        Returns:
            Dict with protein details, or None if not found.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{accession}", params={"format": "json"}, timeout=15
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("UniProt get_protein failed for %s: %s", accession, e)
            return None

        return self._parse_entry(resp.json())

    def get_protein_summary(self, accession: str) -> str:
        """Get a human-readable summary for a protein.

        Args:
            accession: UniProt accession ID.

        Returns:
            Formatted summary string, or empty string on failure.
        """
        protein = self.get_protein(accession)
        if not protein:
            return ""

        parts = [f"**{protein['name']}** ({protein['accession']})"]
        if protein["gene_name"]:
            parts.append(f"Gene: {protein['gene_name']}")
        if protein["organism"]:
            parts.append(f"Organism: {protein['organism']}")
        if protein["function"]:
            parts.append(f"Function: {protein['function']}")
        if protein["sequence_length"]:
            parts.append(f"Sequence length: {protein['sequence_length']} aa")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entry(entry: dict) -> dict:
        """Parse a UniProt JSON entry into a simplified dict."""
        accession = entry.get("primaryAccession", "")

        # Protein name
        protein_desc = entry.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        name = rec_name.get("fullName", {}).get("value", "")
        if not name:
            sub_names = protein_desc.get("submissionNames", [])
            if sub_names:
                name = sub_names[0].get("fullName", {}).get("value", "")

        # Gene name
        genes = entry.get("genes", [])
        gene_name = ""
        if genes:
            gene_name = genes[0].get("geneName", {}).get("value", "")

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "")

        # Function (from comments)
        function = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function = texts[0].get("value", "")
                    break

        # Sequence length
        seq_length = entry.get("sequence", {}).get("length", 0)

        return {
            "accession": accession,
            "name": name,
            "gene_name": gene_name,
            "organism": organism,
            "function": function,
            "sequence_length": seq_length,
        }
