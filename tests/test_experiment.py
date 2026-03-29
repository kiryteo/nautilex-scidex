"""Tests for experiment design module — models, designer, exporter, and API clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from scidex.experiment.models import (
    Control,
    ExperimentProtocol,
    StatisticalPlan,
    Variable,
    VariableType,
)
from scidex.experiment.designer import ExperimentDesigner
from scidex.experiment.exporter import ProtocolExporter
from scidex.experiment.uniprot_client import UniProtClient
from scidex.experiment.pubmed_client import PubMedClient
from scidex.hypothesis.models import Hypothesis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_LLM_RESPONSE = json.dumps(
    {
        "title": "Effect of Compound X on Tumor Growth",
        "objective": "Determine whether Compound X inhibits tumor growth in vitro",
        "background": "Compound X has shown promise in preliminary screens.",
        "variables": [
            {
                "name": "Compound X concentration",
                "variable_type": "independent",
                "description": "Dose of Compound X applied to cells",
                "measurement_method": "Serial dilution",
                "units": "uM",
                "levels": ["0", "1", "10", "100"],
            },
            {
                "name": "Cell viability",
                "variable_type": "dependent",
                "description": "Percentage of live cells after treatment",
                "measurement_method": "MTT assay",
                "units": "%",
                "levels": [],
            },
            {
                "name": "Incubation temperature",
                "variable_type": "controlled",
                "description": "Temperature maintained during experiment",
                "measurement_method": "Thermometer",
                "units": "C",
                "levels": [],
            },
        ],
        "controls": [
            {
                "name": "Vehicle control",
                "description": "DMSO-treated cells",
                "control_type": "vehicle",
                "rationale": "To account for solvent effects",
            },
            {
                "name": "Positive control",
                "description": "Known cytotoxic agent",
                "control_type": "positive",
                "rationale": "To validate assay is working",
            },
        ],
        "statistical_plan": {
            "primary_test": "one-way ANOVA",
            "significance_level": 0.05,
            "power": 0.8,
            "sample_size_per_group": 6,
            "sample_size_justification": "Based on effect size from pilot study",
            "corrections": ["Bonferroni"],
            "secondary_analyses": ["post-hoc Tukey HSD"],
        },
        "methodology": [
            "Seed cells in 96-well plates",
            "Treat with Compound X at specified concentrations",
            "Incubate for 48 hours",
            "Perform MTT assay",
            "Read absorbance at 570 nm",
        ],
        "expected_outcomes": [
            "Dose-dependent decrease in cell viability",
            "IC50 in the low micromolar range",
        ],
        "potential_pitfalls": [
            "Compound X may precipitate at high concentrations",
            "Cell line variation may affect reproducibility",
        ],
        "timeline_weeks": 4,
        "equipment_needed": ["Plate reader", "Cell culture hood"],
        "reagents_needed": ["MTT reagent", "DMSO", "Compound X"],
        "ethical_considerations": ["No animal subjects involved"],
    }
)


@pytest.fixture
def sample_hypothesis():
    return Hypothesis(
        id="hyp-001",
        statement="Compound X inhibits tumor growth by targeting kinase Y",
        hypothesis_type="gap_filling",
        supporting_evidence=["Kinase Y is overexpressed in tumors", "Compound X binds kinase Y"],
        confidence=0.7,
        novelty_score=0.8,
        testability_score=0.9,
        source_papers=["Paper A", "Paper B"],
    )


def _mock_chat(messages, **kwargs) -> str:
    """Return a valid experiment protocol JSON."""
    return SAMPLE_LLM_RESPONSE


@pytest.fixture
def sample_protocol():
    return ExperimentProtocol(
        title="Test Protocol",
        hypothesis_id="hyp-001",
        hypothesis_statement="Test hypothesis statement",
        objective="Test objective",
        background="Test background",
        variables=[
            Variable(
                name="Drug dose",
                variable_type=VariableType.INDEPENDENT,
                description="Amount of drug",
                measurement_method="Pipette",
                units="mg",
                levels=["10", "50", "100"],
            ),
            Variable(
                name="Tumor size",
                variable_type=VariableType.DEPENDENT,
                description="Measured tumor volume",
                measurement_method="Calipers",
                units="mm3",
            ),
        ],
        controls=[
            Control(
                name="Placebo",
                description="Saline injection",
                control_type="negative",
                rationale="Baseline comparison",
            ),
        ],
        statistical_plan=StatisticalPlan(
            primary_test="t-test",
            sample_size_per_group=10,
            sample_size_justification="Power analysis",
        ),
        methodology=["Step 1", "Step 2", "Step 3"],
        expected_outcomes=["Tumor reduction"],
        potential_pitfalls=["Drug toxicity"],
        timeline_weeks=8,
        equipment_needed=["Microscope"],
        reagents_needed=["Saline"],
        ethical_considerations=["Animal welfare"],
        generated_at="2025-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_variable_type_enum(self):
        assert VariableType.INDEPENDENT == "independent"
        assert VariableType.DEPENDENT == "dependent"
        assert VariableType.CONTROLLED == "controlled"
        assert VariableType.CONFOUNDING == "confounding"

    def test_variable_creation(self):
        v = Variable(
            name="Temperature",
            variable_type=VariableType.CONTROLLED,
            description="Ambient temperature",
        )
        assert v.name == "Temperature"
        assert v.variable_type == VariableType.CONTROLLED
        assert v.measurement_method == ""
        assert v.units == ""
        assert v.levels == []

    def test_variable_with_levels(self):
        v = Variable(
            name="Dose",
            variable_type=VariableType.INDEPENDENT,
            description="Drug dosage",
            levels=["low", "medium", "high"],
        )
        assert len(v.levels) == 3

    def test_control_creation(self):
        c = Control(
            name="Vehicle",
            description="DMSO only",
            control_type="vehicle",
            rationale="To exclude solvent effects",
        )
        assert c.name == "Vehicle"
        assert c.control_type == "vehicle"

    def test_statistical_plan_defaults(self):
        sp = StatisticalPlan(
            primary_test="ANOVA",
            sample_size_per_group=20,
            sample_size_justification="Power calculation",
        )
        assert sp.significance_level == 0.05
        assert sp.power == 0.8
        assert sp.corrections == []
        assert sp.secondary_analyses == []

    def test_statistical_plan_custom(self):
        sp = StatisticalPlan(
            primary_test="two-way ANOVA",
            significance_level=0.01,
            power=0.9,
            sample_size_per_group=30,
            sample_size_justification="Large effect expected",
            corrections=["Bonferroni", "Holm"],
            secondary_analyses=["Tukey HSD"],
        )
        assert sp.significance_level == 0.01
        assert len(sp.corrections) == 2

    def test_experiment_protocol_creation(self, sample_protocol):
        assert sample_protocol.title == "Test Protocol"
        assert sample_protocol.hypothesis_id == "hyp-001"
        assert len(sample_protocol.variables) == 2
        assert len(sample_protocol.controls) == 1
        assert sample_protocol.timeline_weeks == 8

    def test_experiment_protocol_defaults(self):
        proto = ExperimentProtocol(
            title="Minimal",
            hypothesis_id="h1",
            hypothesis_statement="stmt",
            objective="obj",
            background="bg",
            variables=[],
            controls=[],
            statistical_plan=StatisticalPlan(
                primary_test="t-test",
                sample_size_per_group=5,
                sample_size_justification="pilot",
            ),
            methodology=[],
            expected_outcomes=[],
            potential_pitfalls=[],
        )
        assert proto.timeline_weeks == 12
        assert proto.equipment_needed == []
        assert proto.reagents_needed == []
        assert proto.ethical_considerations == []
        assert proto.generated_at == ""

    def test_experiment_protocol_model_dump(self, sample_protocol):
        d = sample_protocol.model_dump()
        assert isinstance(d, dict)
        assert d["title"] == "Test Protocol"
        assert isinstance(d["variables"], list)


# ---------------------------------------------------------------------------
# Designer tests
# ---------------------------------------------------------------------------


class TestExperimentDesigner:
    def test_design_returns_protocol(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert isinstance(protocol, ExperimentProtocol)

    def test_design_populates_hypothesis_fields(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert protocol.hypothesis_id == "hyp-001"
        assert "Compound X" in protocol.hypothesis_statement

    def test_design_has_variables(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert len(protocol.variables) == 3
        types = {v.variable_type for v in protocol.variables}
        assert VariableType.INDEPENDENT in types
        assert VariableType.DEPENDENT in types

    def test_design_has_controls(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert len(protocol.controls) == 2

    def test_design_has_statistical_plan(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert protocol.statistical_plan.primary_test == "one-way ANOVA"
        assert protocol.statistical_plan.sample_size_per_group == 6

    def test_design_has_methodology(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert len(protocol.methodology) >= 3

    def test_design_with_domain_context(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis, domain_context="breast cancer cell lines")
        assert isinstance(protocol, ExperimentProtocol)

    def test_design_with_constraints(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(
            sample_hypothesis,
            constraints={"budget": "$5000", "timeline_weeks": 6},
        )
        assert isinstance(protocol, ExperimentProtocol)

    def test_design_sets_generated_at(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        assert protocol.generated_at != ""

    def test_refine_returns_protocol(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        refined = designer.refine(protocol, "Increase sample size to 20")
        assert isinstance(refined, ExperimentProtocol)

    def test_refine_preserves_hypothesis_id(self, sample_hypothesis):
        designer = ExperimentDesigner(chat_fn=_mock_chat)
        protocol = designer.design(sample_hypothesis)
        refined = designer.refine(protocol, "Add a sham control")
        assert refined.hypothesis_id == protocol.hypothesis_id

    def test_design_handles_empty_llm_response(self, sample_hypothesis):
        def bad_chat(messages, **kwargs):
            return "not json at all"

        designer = ExperimentDesigner(chat_fn=bad_chat)
        protocol = designer.design(sample_hypothesis)
        # Should not raise — produces a protocol with defaults
        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.title == "Untitled Experiment"

    def test_design_handles_malformed_variable(self, sample_hypothesis):
        """LLM returns a variable with an invalid type — should be skipped."""
        bad_response = json.dumps(
            {
                "title": "Test",
                "variables": [
                    {"name": "Good", "variable_type": "independent", "description": "ok"},
                    {"name": "Bad", "variable_type": "INVALID_TYPE", "description": "bad"},
                ],
                "controls": [],
                "statistical_plan": {
                    "primary_test": "t-test",
                    "sample_size_per_group": 5,
                    "sample_size_justification": "guess",
                },
                "methodology": [],
                "expected_outcomes": [],
                "potential_pitfalls": [],
            }
        )

        def chat_fn(messages, **kwargs):
            return bad_response

        designer = ExperimentDesigner(chat_fn=chat_fn)
        protocol = designer.design(sample_hypothesis)
        # The bad variable should be skipped
        assert len(protocol.variables) == 1
        assert protocol.variables[0].name == "Good"


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------


class TestProtocolExporter:
    def test_to_markdown_has_title(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "# Test Protocol" in md

    def test_to_markdown_has_hypothesis(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "Test hypothesis statement" in md

    def test_to_markdown_has_variables_table(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "| Name | Type |" in md
        assert "Drug dose" in md
        assert "independent" in md

    def test_to_markdown_has_controls_table(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "Placebo" in md
        assert "negative" in md

    def test_to_markdown_has_statistical_plan(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "t-test" in md
        assert "10" in md  # sample size

    def test_to_markdown_has_methodology(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "Step 1" in md
        assert "Step 2" in md

    def test_to_markdown_has_equipment(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "Microscope" in md

    def test_to_markdown_has_ethical_considerations(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "Animal welfare" in md

    def test_to_markdown_has_variable_levels(self, sample_protocol):
        exporter = ProtocolExporter()
        md = exporter.to_markdown(sample_protocol)
        assert "10, 50, 100" in md

    def test_to_dict_returns_dict(self, sample_protocol):
        exporter = ProtocolExporter()
        d = exporter.to_dict(sample_protocol)
        assert isinstance(d, dict)
        assert d["title"] == "Test Protocol"

    def test_to_dict_is_json_serializable(self, sample_protocol):
        exporter = ProtocolExporter()
        d = exporter.to_dict(sample_protocol)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_markdown_empty_optional_sections(self):
        """Protocol with no equipment/reagents/ethics should not have those sections."""
        proto = ExperimentProtocol(
            title="Minimal",
            hypothesis_id="h1",
            hypothesis_statement="stmt",
            objective="obj",
            background="bg",
            variables=[],
            controls=[],
            statistical_plan=StatisticalPlan(
                primary_test="t-test",
                sample_size_per_group=5,
                sample_size_justification="pilot",
            ),
            methodology=[],
            expected_outcomes=[],
            potential_pitfalls=[],
        )
        exporter = ProtocolExporter()
        md = exporter.to_markdown(proto)
        assert "Equipment Needed" not in md
        assert "Reagents Needed" not in md
        assert "Ethical Considerations" not in md


# ---------------------------------------------------------------------------
# UniProt client tests
# ---------------------------------------------------------------------------


class TestUniProtClient:
    def test_search_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "primaryAccession": "P04637",
                    "proteinDescription": {
                        "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}},
                    },
                    "genes": [{"geneName": {"value": "TP53"}}],
                    "organism": {"scientificName": "Homo sapiens"},
                    "comments": [
                        {
                            "commentType": "FUNCTION",
                            "texts": [{"value": "Acts as a tumor suppressor"}],
                        }
                    ],
                    "sequence": {"length": 393},
                }
            ]
        }
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        results = client.search("TP53")
        assert len(results) == 1
        assert results[0]["accession"] == "P04637"
        assert results[0]["gene_name"] == "TP53"
        assert results[0]["organism"] == "Homo sapiens"
        assert "tumor suppressor" in results[0]["function"]

    def test_search_empty(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        results = client.search("nonexistent_protein_xyz")
        assert results == []

    def test_search_network_error(self, monkeypatch):
        import requests as req

        def raise_error(*a, **kw):
            raise req.ConnectionError("Network error")

        monkeypatch.setattr("scidex.experiment.uniprot_client.requests.get", raise_error)

        client = UniProtClient()
        results = client.search("TP53")
        assert results == []

    def test_get_protein_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "primaryAccession": "P04637",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}},
            },
            "genes": [{"geneName": {"value": "TP53"}}],
            "organism": {"scientificName": "Homo sapiens"},
            "comments": [],
            "sequence": {"length": 393},
        }
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        result = client.get_protein("P04637")
        assert result is not None
        assert result["accession"] == "P04637"

    def test_get_protein_not_found(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 404
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        result = client.get_protein("NONEXISTENT")
        assert result is None

    def test_get_protein_summary(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "primaryAccession": "P04637",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "p53"}},
            },
            "genes": [{"geneName": {"value": "TP53"}}],
            "organism": {"scientificName": "Homo sapiens"},
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": "Tumor suppressor"}]},
            ],
            "sequence": {"length": 393},
        }
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        summary = client.get_protein_summary("P04637")
        assert "TP53" in summary
        assert "Homo sapiens" in summary

    def test_get_protein_summary_not_found(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 404
        monkeypatch.setattr(
            "scidex.experiment.uniprot_client.requests.get", lambda *a, **kw: mock_response
        )

        client = UniProtClient()
        summary = client.get_protein_summary("NONEXISTENT")
        assert summary == ""


# ---------------------------------------------------------------------------
# PubMed client tests
# ---------------------------------------------------------------------------

SAMPLE_PUBMED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>CRISPR-based gene therapy for cancer</ArticleTitle>
        <Abstract>
          <AbstractText>We demonstrate CRISPR-based gene editing for cancer treatment.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <ArticleTitle>RNA-seq analysis of tumor samples</ArticleTitle>
        <Abstract>
          <AbstractText>RNA-seq reveals novel biomarkers.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Lee</LastName></Author>
        </AuthorList>
        <Journal>
          <Title>Cell</Title>
          <JournalIssue>
            <PubDate><Year>2023</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedClient:
    def test_search_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": ["12345678", "87654321"]}}
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        pmids = client.search("CRISPR cancer")
        assert pmids == ["12345678", "87654321"]

    def test_search_empty(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        pmids = client.search("zzzznonexistent")
        assert pmids == []

    def test_search_network_error(self, monkeypatch):
        import requests as req

        def raise_error(*a, **kw):
            raise req.ConnectionError("Network error")

        monkeypatch.setattr("scidex.experiment.pubmed_client.requests.get", raise_error)

        client = PubMedClient()
        pmids = client.search("CRISPR")
        assert pmids == []

    def test_fetch_abstracts_success(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = SAMPLE_PUBMED_XML
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        papers = client.fetch_abstracts(["12345678", "87654321"])
        assert len(papers) == 2
        assert papers[0]["pmid"] == "12345678"
        assert papers[0]["title"] == "CRISPR-based gene therapy for cancer"
        assert "gene editing" in papers[0]["abstract"]
        assert papers[0]["authors"] == ["John Smith", "Jane Doe"]
        assert papers[0]["journal"] == "Nature Medicine"
        assert papers[0]["year"] == "2024"

    def test_fetch_abstracts_author_last_name_only(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = SAMPLE_PUBMED_XML
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        papers = client.fetch_abstracts(["87654321"])
        # Second paper has only LastName for one author
        assert "Lee" in papers[1]["authors"]

    def test_fetch_abstracts_empty_pmids(self, monkeypatch):
        client = PubMedClient()
        papers = client.fetch_abstracts([])
        assert papers == []

    def test_fetch_abstracts_bad_xml(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "this is not xml"
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        papers = client.fetch_abstracts(["12345678"])
        assert papers == []

    def test_search_and_fetch(self, monkeypatch):
        call_count = {"n": 0}

        def mock_get(*a, **kw):
            call_count["n"] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            if call_count["n"] == 1:
                # search call
                mock_resp.json = MagicMock(return_value={"esearchresult": {"idlist": ["12345678"]}})
            else:
                # fetch call
                mock_resp.text = SAMPLE_PUBMED_XML
            return mock_resp

        monkeypatch.setattr("scidex.experiment.pubmed_client.requests.get", mock_get)

        client = PubMedClient()
        papers = client.search_and_fetch("CRISPR", max_results=1)
        assert len(papers) >= 1

    def test_search_and_fetch_no_results(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        monkeypatch.setattr(
            "scidex.experiment.pubmed_client.requests.get", lambda *a, **kw: mock_response
        )

        client = PubMedClient()
        papers = client.search_and_fetch("nonexistent")
        assert papers == []
