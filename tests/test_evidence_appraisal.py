from datetime import UTC, datetime
from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.evidence import EvidenceBundle, normalize_pubmed, normalize_europe_pmc
from medic.ledger import AppendOnlyJsonlLedger
from medic.providers import ApprovedGateway, RecordedGateway


class EvidenceAppraisalTests(unittest.TestCase):
    def test_doi_bridge_deduplicates_records_with_different_available_ids(self):
        now = datetime.now(UTC)
        first = normalize_pubmed({"uid": "123", "title": "Study", "elocationid": "doi: 10.1/test"}, now)
        second = normalize_europe_pmc({"id": "PMC1", "title": "Study", "doi": "https://doi.org/10.1/TEST"}, now)
        self.assertEqual(len(EvidenceBundle.deduplicated((first, second)).cards), 1)

    def test_appraisal_requires_exact_source_quotes_and_preserves_abstract_scope(self):
        from medic.evidence_appraisal import EvidenceAppraiser
        now = datetime.now(UTC)
        card = normalize_pubmed({"uid": "123", "title": "Study", "_abstract": "Twenty adults were followed for one year."}, now)
        payload = {"fields": [{"field": "population", "value": "20 adults", "quote": "Twenty adults"}],
                   "applicability": ["Single-year followup does not establish long-term prognosis."],
                   "limitations": ["Abstract only"], "missing_fields": ["comparator", "effect_estimates"]}
        with tempfile.TemporaryDirectory() as temp:
            raw = RecordedGateway("anthropic", "model", [payload])
            appraiser = EvidenceAppraiser(ApprovedGateway(raw), AppendOnlyJsonlLedger(Path(temp) / "ledger.jsonl"))
            estimate = ModelRegistry.default().estimate_deliberation_cost(100).approve()
            result = appraiser.appraise(card, {"confirmed_facts": []}, estimate)
            self.assertEqual(result.population, "20 adults")
            self.assertEqual(result.access_level, "ABSTRACT")
            self.assertEqual(result.field_support[0].quote, "Twenty adults")
            self.assertEqual(appraiser.appraise(card, {"confirmed_facts": []}, estimate), result)
            self.assertEqual(raw.call_count, 1)
            bad = {**payload, "fields": [{"field": "effect_estimates", "value": "90%", "quote": "Ninety percent"}]}
            other = EvidenceAppraiser(ApprovedGateway(RecordedGateway("anthropic", "different", [bad])), appraiser.ledger)
            with self.assertRaisesRegex(ValueError, "quote"):
                other.appraise(card, {}, estimate)
