import unittest
import httpx


class SearchSafetyTests(unittest.TestCase):
    def test_workflow_stops_before_external_search_until_queries_reviewed(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from medic.workflow import ResearchWorkflow, ResearchRunFailed
        from medic.config import ModelRegistry
        from medic.research_contracts import QuestionSet
        import tests.test_tribunal as fixtures
        class Retriever:
            sources = ()
            calls = 0
            def search(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError('external search must not run')
        with tempfile.TemporaryDirectory() as temp:
            _, gateways = fixtures.gateways()
            with patch('medic.workflow.LiteratureSearchPlanner.plan', return_value=SimpleNamespace(queries=('ankle',))):
                with self.assertRaises(ResearchRunFailed):
                    retriever = Retriever()
                    ResearchWorkflow(Path(temp), gateways, retriever).run(case_id='case', patient_context={},
                        question_set=QuestionSet.for_comprehensive_review('case'),
                        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve(), run_id='review-search')
                self.assertEqual(retriever.calls, 0)

    def test_identifiers_blocked_before_search(self):
        from medic.search_transport import validate_query
        for query in ('ankle user@example.com', 'ankle C:\\records', 'ankle 010-1234-5678', 'patient 123456-1234567'):
            with self.assertRaises(ValueError):
                validate_query(query)
        self.assertEqual(validate_query('ankle fracture systematic review'), 'ankle fracture systematic review')

    def test_europe_pmc_paginates_with_cursor_and_reports_limit(self):
        from medic.evidence import EuropePmcSource
        seen = []
        def respond(request):
            cursor = request.url.params.get('cursorMark')
            seen.append(cursor)
            n = '1' if cursor == '*' else '2'
            return httpx.Response(200, json={'hitCount': 5, 'nextCursorMark': n,
                'resultList': {'result': [{'id': n, 'pmid': n, 'title': 'Ankle', 'abstractText': 'abstract'}]}})
        source = EuropePmcSource(client=httpx.Client(transport=httpx.MockTransport(respond)), limit=2)
        self.assertEqual(len(source.search('ankle')), 2)
        self.assertEqual(seen, ['*', '1'])
        self.assertTrue(source.last_search['truncated'])
