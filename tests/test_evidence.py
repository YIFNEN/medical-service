from __future__ import annotations

import unittest
from datetime import UTC, datetime

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import EvidenceCard


class EvidenceTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_pubmed_and_europe_pmc_by_pmid(self):
        from medic.evidence import EvidenceBundle, normalize_europe_pmc, normalize_pubmed

        retrieved = datetime.now(UTC)
        pubmed = normalize_pubmed(
            {
                "uid": "123",
                "title": "Ankle outcomes",
                "elocationid": "doi: 10.1/test",
                "pubtype": ["Meta-Analysis"],
                "_abstract": "PTOA occurred in 20 percent of participants.",
            },
            retrieved,
        )
        europe = normalize_europe_pmc(
            {
                "pmid": "123",
                "title": "Ankle outcomes",
                "doi": "10.1/test",
                "pubTypeList": {"pubType": ["meta-analysis"]},
            },
            retrieved,
        )

        bundle = EvidenceBundle.deduplicated((pubmed, europe))

        self.assertEqual(len(bundle.cards), 1)
        self.assertEqual(bundle.cards[0].pmid, "123")
        self.assertIn("20 percent", bundle.cards[0].findings[0])

    def test_normalizes_clinical_trial(self):
        from medic.evidence import normalize_clinical_trial

        card = normalize_clinical_trial(
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "briefTitle": "Ankle rehabilitation",
                    },
                    "designModule": {"studyType": "INTERVENTIONAL"},
                    "contactsLocationsModule": {},
                }
            },
            datetime.now(UTC),
        )

        self.assertEqual(card.nct_id, "NCT00000001")
        self.assertIn("clinicaltrials.gov", str(card.url))

    def test_citation_gate_blocks_unknown_evidence_id(self):
        from medic.evidence import CitationGate

        card = EvidenceCard(
            evidence_id="ev-1",
            source="PUBMED",
            title="Study",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            retrieved_at=datetime.now(UTC),
            study_design="cohort",
            population="adults",
            findings=("finding",),
            limitations=("limitation",),
        )
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            CitationGate((card,)).validate(("ev-1", "hallucinated"))


    def test_official_ankle_guidelines_are_normalized_as_guideline_cards(self):
        from medic.evidence import OfficialAnkleGuidelineSource

        import httpx
        client = httpx.Client(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, text="<main>Official guideline text</main>")))
        cards = OfficialAnkleGuidelineSource(client=client).search("ankle fracture arthritis")
        self.assertGreaterEqual(len(cards), 2)
        self.assertTrue(all(card.source == "GUIDELINE" for card in cards))
        self.assertTrue(any("nice.org.uk" in str(card.url) for card in cards))
    def test_three_models_independently_plan_picots_and_query_union(self):
        from medic.evidence import LiteratureSearchPlanner
        from medic.research_contracts import QuestionSet

        gateways = {}
        for provider, query in (
            ("openai", "ankle fracture osteoarthritis meta-analysis"),
            ("anthropic", "trimalleolar fracture malunion systematic review"),
            ("google", "ankle ORIF syndesmosis outcomes"),
        ):
            gateways[provider] = ApprovedGateway(RecordedGateway(provider, f"m-{provider}", [{
                "provider": "hallucinated",
                "model": "hallucinated",
                "provider_response_id": "hallucinated",
                "_response_id": f"actual-{provider}",
                "picots": {"population": "adult ankle ORIF", "intervention": "follow-up",
                           "comparator": "alternative management", "outcomes": ["PTOA"],
                           "timing": "longitudinal", "study_designs": ["systematic review"]},
                "queries": [query, "shared ankle query"],
            }]))
        result = LiteratureSearchPlanner(gateways).plan(
            QuestionSet.for_comprehensive_review("case-1", "관절염 위험?"),
            {"confirmed_facts": ["right ankle ORIF"]},
            ModelRegistry.default().estimate_tribunal_cost(100).approve(),
        )

        self.assertEqual(len(result.plans), 3)
        self.assertEqual([plan.provider for plan in result.plans], ["openai", "anthropic", "google"])
        self.assertEqual([plan.provider_response_id for plan in result.plans], ["actual-openai", "actual-anthropic", "actual-google"])
        self.assertEqual(len(result.queries), 4)
        for provider, gateway in gateways.items():
            for other in set(gateways) - {provider}:
                self.assertNotIn(f"r-{other}", gateway.inner.prompts[0])
if __name__ == "__main__":
    unittest.main()
