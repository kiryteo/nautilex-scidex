"""Extract entities from paper metadata and abstracts using regex patterns.

No LLM or spaCy dependency — uses structured metadata and curated regex patterns
for scientific entity recognition.
"""

from __future__ import annotations

import re
import hashlib
from typing import Sequence

from scidex.knowledge_graph.models import Entity

# ---------------------------------------------------------------------------
# Curated pattern lists
# ---------------------------------------------------------------------------

# Common scientific methods (case-insensitive matching)
_METHODS = [
    "PCR",
    "qPCR",
    "RT-PCR",
    "CRISPR",
    "CRISPR-Cas9",
    "RNA-seq",
    "RNA-Seq",
    "scRNA-seq",
    "ChIP-seq",
    "ATAC-seq",
    "Hi-C",
    "mass spectrometry",
    "flow cytometry",
    "Western blot",
    "ELISA",
    "immunohistochemistry",
    "immunofluorescence",
    "confocal microscopy",
    "electron microscopy",
    "cryo-EM",
    "X-ray crystallography",
    "NMR spectroscopy",
    "FACS",
    "microarray",
    "deep learning",
    "machine learning",
    "random forest",
    "neural network",
    "convolutional neural network",
    "transformer",
    "single-cell RNA sequencing",
    "whole genome sequencing",
    "WGS",
    "genome-wide association study",
    "GWAS",
    "meta-analysis",
    "systematic review",
    "clinical trial",
    "randomized controlled trial",
    "Monte Carlo",
    "Bayesian",
    "logistic regression",
    "Cox regression",
    "Kaplan-Meier",
    "PCA",
    "t-SNE",
    "UMAP",
]

# Common diseases (case-insensitive)
_DISEASES = [
    "cancer",
    "breast cancer",
    "lung cancer",
    "prostate cancer",
    "colorectal cancer",
    "pancreatic cancer",
    "melanoma",
    "leukemia",
    "lymphoma",
    "glioblastoma",
    "hepatocellular carcinoma",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Huntington's disease",
    "diabetes",
    "type 2 diabetes",
    "type 1 diabetes",
    "hypertension",
    "atherosclerosis",
    "heart failure",
    "asthma",
    "COPD",
    "pneumonia",
    "COVID-19",
    "SARS-CoV-2",
    "influenza",
    "HIV",
    "AIDS",
    "tuberculosis",
    "malaria",
    "obesity",
    "depression",
    "schizophrenia",
    "autism",
    "multiple sclerosis",
    "rheumatoid arthritis",
    "lupus",
    "Crohn's disease",
    "ulcerative colitis",
    "fibrosis",
    "sepsis",
    "stroke",
    "epilepsy",
    "ALS",
]

# Protein suffix patterns
_PROTEIN_SUFFIXES = re.compile(
    r"\b([A-Z][a-z]*(?:ase|in|gen|sin|lin|tin|rin|nin|ein|din|kin|ine))\b"
)

# Gene name pattern: 2-6 uppercase letters optionally followed by digits
_GENE_PATTERN = re.compile(r"\b([A-Z]{2,6}[0-9]{0,2})\b")

# Words to exclude from gene matching (common English/scientific abbreviations)
_GENE_EXCLUDE = {
    "THE",
    "AND",
    "FOR",
    "NOT",
    "BUT",
    "ARE",
    "WAS",
    "HAS",
    "HAD",
    "HIS",
    "HER",
    "ITS",
    "OUR",
    "WHO",
    "HOW",
    "ALL",
    "CAN",
    "MAY",
    "USE",
    "USED",
    "NEW",
    "TWO",
    "ONE",
    "SET",
    "LET",
    "SAY",
    "SAW",
    "OLD",
    "END",
    "FAR",
    "GET",
    "GOT",
    "RUN",
    "MAN",
    "TRY",
    "ASK",
    "OWN",
    "TOO",
    "ANY",
    "DAY",
    "WAY",
    "NOW",
    "EACH",
    "MAKE",
    "LIKE",
    "LONG",
    "LOOK",
    "MANY",
    "THEN",
    "THEM",
    "THAN",
    "BEEN",
    "HAVE",
    "SAID",
    "WILL",
    "INTO",
    "TIME",
    "VERY",
    "WHEN",
    "COME",
    "COULD",
    "MORE",
    "MADE",
    "AFTER",
    "ALSO",
    "DID",
    "FROM",
    "MOST",
    "WITH",
    "THIS",
    "THAT",
    "WHAT",
    "OVER",
    "SUCH",
    "YOUR",
    "WERE",
    "SOME",
    "ONLY",
    "JUST",
    "MUCH",
    "KNOW",
    "TAKE",
    "THAN",
    "HERE",
    "WELL",
    "BACK",
    "YEAR",
    "LAST",
    "GOOD",
    "HIGH",
    "BOTH",
    "SAME",
    "RNA",
    "DNA",
    "USA",
    "UK",
    "EU",
    "WHO",
    "NIH",
    "FDA",
    "IEEE",
    "RESULTS",
    "METHODS",
    "STUDY",
    "DATA",
    "USING",
    "BASED",
    "SHOWED",
    "PAPER",
    "FOUND",
    "FIRST",
    "THESE",
    "CELL",
    "CELLS",
    "TYPE",
    "HUMAN",
    "MODEL",
    "RISK",
    "GENE",
    "GENES",
    "DRUG",
    "DRUGS",
    "EFFECT",
    "GROUP",
    "LEVEL",
    "CASE",
    "ROLE",
    "RATE",
    "DOSE",
    "SIZE",
    "LOSS",
    "LACK",
    "NEED",
    "WORK",
    "PART",
    "FORM",
    "AREA",
    "LINE",
    "NAME",
    "BODY",
    "LEAD",
    "TERM",
    "FACT",
    "IDEA",
    "TEST",
    "BEST",
    "TRUE",
    "REAL",
    "ABLE",
    "ACID",
}


def _make_id(entity_type: str, name: str) -> str:
    """Deterministic ID from type + lowercased name."""
    raw = f"{entity_type}:{name.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class EntityExtractor:
    """Extract structured entities from paper metadata and abstracts."""

    def __init__(self) -> None:
        # Pre-compile method and disease patterns for fast matching
        self._method_patterns: list[tuple[re.Pattern, str]] = []
        for m in sorted(_METHODS, key=len, reverse=True):
            pat = re.compile(r"\b" + re.escape(m) + r"\b", re.IGNORECASE)
            self._method_patterns.append((pat, m))

        self._disease_patterns: list[tuple[re.Pattern, str]] = []
        for d in sorted(_DISEASES, key=len, reverse=True):
            pat = re.compile(r"\b" + re.escape(d) + r"\b", re.IGNORECASE)
            self._disease_patterns.append((pat, d))

    # ------------------------------------------------------------------
    # Structured extraction from paper metadata
    # ------------------------------------------------------------------

    def extract_from_paper(self, paper_metadata: dict) -> list[Entity]:
        """Extract entities from structured paper metadata (no LLM needed).

        Extracts: paper entity, author entities, venue entity.
        """
        entities: list[Entity] = []
        paper_id = paper_metadata.get("paperId", paper_metadata.get("paper_id", ""))
        title = paper_metadata.get("title", "")

        # Paper entity
        if paper_id and title:
            entities.append(
                Entity(
                    id=paper_id,
                    name=title,
                    entity_type="paper",
                    properties={
                        "year": paper_metadata.get("year"),
                        "citation_count": paper_metadata.get(
                            "citationCount", paper_metadata.get("citation_count", 0)
                        ),
                    },
                )
            )

        # Author entities
        for author in paper_metadata.get("authors", []):
            if isinstance(author, dict):
                name = author.get("name", "")
                author_id = author.get("authorId", "")
            elif isinstance(author, str):
                name = author
                author_id = ""
            else:
                continue
            if not name:
                continue
            entities.append(
                Entity(
                    id=author_id or _make_id("author", name),
                    name=name,
                    entity_type="author",
                    properties={},
                )
            )

        # Venue entity
        venue = paper_metadata.get("venue") or paper_metadata.get("journal", {})
        if isinstance(venue, dict):
            venue_name = venue.get("name", "")
        elif isinstance(venue, str):
            venue_name = venue
        else:
            venue_name = ""
        if venue_name:
            entities.append(
                Entity(
                    id=_make_id("venue", venue_name),
                    name=venue_name,
                    entity_type="venue",
                    properties={},
                )
            )

        return entities

    # ------------------------------------------------------------------
    # Regex-based extraction from abstract text
    # ------------------------------------------------------------------

    def extract_from_abstract(self, abstract: str) -> list[Entity]:
        """Extract scientific entities from abstract text using regex patterns.

        Detects: genes, proteins, methods, diseases.
        """
        if not abstract:
            return []

        entities: list[Entity] = []
        seen_names: set[str] = set()

        # Methods
        for pat, method_name in self._method_patterns:
            match = pat.search(abstract)
            if match:
                key = method_name.lower()
                if key not in seen_names:
                    seen_names.add(key)
                    # Extract context: surrounding sentence fragment
                    start = max(0, match.start() - 40)
                    end = min(len(abstract), match.end() + 40)
                    context = abstract[start:end].strip()
                    entities.append(
                        Entity(
                            id=_make_id("method", method_name),
                            name=method_name,
                            entity_type="method",
                            properties={},
                            mentions=[context],
                        )
                    )

        # Diseases
        for pat, disease_name in self._disease_patterns:
            match = pat.search(abstract)
            if match:
                key = disease_name.lower()
                if key not in seen_names:
                    seen_names.add(key)
                    start = max(0, match.start() - 40)
                    end = min(len(abstract), match.end() + 40)
                    context = abstract[start:end].strip()
                    entities.append(
                        Entity(
                            id=_make_id("disease", disease_name),
                            name=disease_name,
                            entity_type="disease",
                            properties={},
                            mentions=[context],
                        )
                    )

        # Genes (uppercase 2-6 letter patterns)
        for match in _GENE_PATTERN.finditer(abstract):
            gene_name = match.group(1)
            if gene_name in _GENE_EXCLUDE:
                continue
            if len(gene_name) < 2:
                continue
            key = gene_name.lower()
            if key not in seen_names:
                seen_names.add(key)
                start = max(0, match.start() - 40)
                end = min(len(abstract), match.end() + 40)
                context = abstract[start:end].strip()
                entities.append(
                    Entity(
                        id=_make_id("gene", gene_name),
                        name=gene_name,
                        entity_type="gene",
                        properties={},
                        mentions=[context],
                    )
                )

        # Proteins (suffix-based)
        for match in _PROTEIN_SUFFIXES.finditer(abstract):
            protein_name = match.group(1)
            key = protein_name.lower()
            if key not in seen_names:
                seen_names.add(key)
                start = max(0, match.start() - 40)
                end = min(len(abstract), match.end() + 40)
                context = abstract[start:end].strip()
                entities.append(
                    Entity(
                        id=_make_id("protein", protein_name),
                        name=protein_name,
                        entity_type="protein",
                        properties={},
                        mentions=[context],
                    )
                )

        return entities
