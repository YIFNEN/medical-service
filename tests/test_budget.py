import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class BudgetTests(unittest.TestCase):
    def test_reservation_overrun_blocks_new_calls_after_restart(self):
        from medic.budget import BudgetLedger, BudgetExceeded
        from medic.ledger import AppendOnlyJsonlLedger
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            budget = BudgetLedger(ledger, 'approval', 10.0, {})
            budget.reserve('a', 0.1)
            with self.assertRaises(BudgetExceeded):
                budget.settle('a', 0.2)
            resumed = BudgetLedger(ledger, 'approval', 10.0, {})
            with self.assertRaisesRegex(BudgetExceeded, 'reservation'):
                resumed.reserve('b', 0.1)
            self.assertAlmostEqual(resumed.committed(), 0.2)
            self.assertEqual(sum(r['kind'] == 'BUDGET_RESERVED' for r in ledger.records()), 1)

    def test_unapproved_estimate_does_not_reserve_funds(self):
        from medic.call_cache import cached_generate
        from medic.config import ModelRegistry
        from medic.ledger import AppendOnlyJsonlLedger
        from medic.providers import ApprovedGateway, RecordedGateway
        from medic.research_contracts import StrictFrozenModel
        class Output(StrictFrozenModel):
            value: str
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            estimate = ModelRegistry.default().estimate_tribunal_cost(100).model_copy(update={'budget_usd': 1.0})
            with self.assertRaises(PermissionError):
                cached_generate(ledger, ApprovedGateway(RecordedGateway('openai','test',[])), scope='a',
                    prompt='test', stage='TEST', output_model=Output, estimate=estimate, max_output_tokens=1)
            self.assertFalse(any(r['kind'] == 'BUDGET_RESERVED' for r in ledger.records()))

    def test_cached_call_reserves_and_does_not_recharge_on_resume(self):
        from medic.budget import BudgetLedger
        from medic.call_cache import cached_generate
        from medic.config import ModelRegistry
        from medic.ledger import AppendOnlyJsonlLedger
        from medic.providers import ApprovedGateway, RecordedGateway
        from medic.research_contracts import StrictFrozenModel
        class Output(StrictFrozenModel):
            value: str
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            registry = ModelRegistry.default()
            estimate = registry.estimate_tribunal_cost(100).approve().model_copy(update={'budget_usd': 2.0})
            raw = RecordedGateway('openai', 'model', [{'value': 'ok'}])
            gateway = ApprovedGateway(raw)
            args = dict(scope='test', stage='TEST', prompt='test', output_model=Output, estimate=estimate, max_output_tokens=10)
            self.assertEqual(cached_generate(ledger, gateway, **args).value.value, 'ok')
            before = len(ledger.records())
            cached_generate(ledger, gateway, **args)
            self.assertEqual(before, len(ledger.records()))
            self.assertEqual(raw.call_count, 1)
            self.assertTrue(any(r['kind'] == 'BUDGET_RESERVED' for r in ledger.records()))

    def test_reservations_survive_restart_and_settle_once(self):
        from medic.budget import BudgetLedger, BudgetExceeded
        from medic.ledger import AppendOnlyJsonlLedger
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            budget = BudgetLedger(ledger, 'approval', 1.0, {'price': 2})
            budget.reserve('a', 0.7)
            resumed = BudgetLedger(ledger, 'approval', 1.0, {'price': 2})
            with self.assertRaises(BudgetExceeded):
                resumed.reserve('b', 0.4)
            resumed.settle('a', 0.2)
            resumed.settle('a', 0.2)
            resumed.reserve('b', 0.4)
            self.assertAlmostEqual(resumed.committed(), 0.6)
            with self.assertRaises(ValueError):
                BudgetLedger(ledger, 'approval', 2.0, {'price': 2})

    def test_concurrent_reservations_cannot_overspend(self):
        from medic.budget import BudgetLedger, BudgetExceeded
        from medic.ledger import AppendOnlyJsonlLedger
        with tempfile.TemporaryDirectory() as temp:
            budget = BudgetLedger(AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl'), 'a', 1.0, {})
            def reserve(index):
                try:
                    budget.reserve(str(index), 0.6)
                    return True
                except BudgetExceeded:
                    return False
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(sum(pool.map(reserve, range(4))), 1)

    def test_unbounded_images_and_invalid_amounts_fail_closed(self):
        from medic.budget import BudgetLedger, request_bound
        from medic.config import ModelRegistry
        from medic.ledger import AppendOnlyJsonlLedger
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            with self.assertRaises(ValueError):
                BudgetLedger(ledger, 'a', float('nan'), {})
            with self.assertRaisesRegex(ValueError, 'image'):
                request_bound(ModelRegistry.default(), 'openai', 'x', {}, 10, (Path('image'),))
