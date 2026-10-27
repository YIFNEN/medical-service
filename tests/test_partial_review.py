import tempfile
import unittest
from pathlib import Path

from medic.extraction import ClinicalExtractionService, ExtractionResult, FactReconciliation, ReconciledFact, FactKey


class PartialReviewTests(unittest.TestCase):
    def test_reverting_to_previous_decision_remains_latest(self):
        from medic.cli import _latest_context
        from medic.ledger import AppendOnlyJsonlLedger

        with tempfile.TemporaryDirectory() as temp:
            fact = ReconciledFact(fact_key=FactKey.UNION, proposed_value='A', source_locators=('a',),
                                  supporting_providers=('openai',), conflict=True,
                                  requires_user_confirmation=True, verification_note='check')
            result = ExtractionResult(extraction_id='ex', intake_id='in', case_id='case', independent_batches=(),
                                      reconciliation=FactReconciliation(provider_response_id='r', facts=(fact,), uncertainties=()))
            service = ClinicalExtractionService(Path(temp), dict.fromkeys(('openai', 'google', 'anthropic')))
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            for value in ('A', 'B', 'A'):
                service.confirm(result, {fact.fact_id: value})
            self.assertEqual(service.saved_decisions(result)[fact.fact_id], 'A')
            self.assertEqual(_latest_context(ledger, 'case')['fact_details'][0]['value'], 'A')
            count = len(ledger.records())
            service.confirm(result, {fact.fact_id: 'A'})
            self.assertEqual(len(ledger.records()), count)
            service.save_decision(result, fact.fact_id, None, withdraw=True)
            service.confirm(result, {fact.fact_id: 'A'})
            self.assertEqual(_latest_context(ledger, 'case')['fact_details'][0]['value'], 'A')

    def test_partial_decisions_and_withdrawal_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            fact = ReconciledFact(fact_key=FactKey.UNION, proposed_value='uncertain', source_locators=('a',),
                                  supporting_providers=('openai',), conflict=True,
                                  requires_user_confirmation=True, verification_note='check')
            result = ExtractionResult(extraction_id='ex', intake_id='in', case_id='case', independent_batches=(),
                                      reconciliation=FactReconciliation(provider_response_id='r', facts=(fact,), uncertainties=()))
            service = ClinicalExtractionService(Path(temp), dict.fromkeys(('openai','google','anthropic')))
            service.save_decision(result, fact.fact_id, 'source wording')
            self.assertFalse(service.pending_facts(result))
            service.confirm(result, {})
            service.save_decision(result, fact.fact_id, None, withdraw=True)
            self.assertEqual(len(service.pending_facts(result)), 1)
            from medic.cli import _latest_context
            from medic.ledger import AppendOnlyJsonlLedger
            self.assertFalse(_latest_context(AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl'), 'case')['fact_details'])
            service.save_decision(result, fact.fact_id, None)
            self.assertFalse(service.pending_facts(result))
            self.assertFalse(service.confirm(result, {})['fact_details'])
