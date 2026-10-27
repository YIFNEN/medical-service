import tempfile
import hashlib
import unittest
from pathlib import Path

import httpx
import pymupdf

from medic.ledger import AppendOnlyJsonlLedger
from medic.literature import FullTextResolver
from tests.test_decision_literature import card


class FullTextRecoveryTests(unittest.TestCase):
    def test_corrupt_local_pdf_falls_back_to_pmc(self):
        xml = '<article><front><article-meta><article-id pub-id-type="pmc">123</article-id></article-meta></front><body><sec>Methods: adults.</sec></body></article>'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.pdf'
            data = b'not a PDF'
            path.write_bytes(data)
            item = card().model_copy(update={'fulltext_required': True})
            resolver = FullTextResolver(client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=xml))), local_papers={item.evidence_id:
                {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'identity_confirmed': True}})
            self.assertEqual(resolver.fetch(item).access_level, 'FULL_TEXT')

    def test_legacy_failure_cache_is_preserved_until_explicit_retry(self):
        from medic.ledger import deterministic_id
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            item = card().model_copy(update={'fulltext_required': True})
            key = deterministic_id('fulltext/4', {'id': item.evidence_id, 'binding': None,
                'pmcid': item.pmcid, 'url': item.fulltext_url, 'doi': item.doi, 'title': item.title})
            old = item.model_copy(update={'fulltext_status': 'UNAVAILABLE: previous failure'})
            ledger.append_once('FULLTEXT_FETCHED', key, {'card': old.model_dump(mode='json')})
            calls = []
            def respond(request):
                calls.append(request)
                return httpx.Response(404)
            resolver = FullTextResolver(client=httpx.Client(transport=httpx.MockTransport(respond)), ledger=ledger)
            self.assertEqual(resolver.fetch(item).fulltext_status, old.fulltext_status)
            self.assertFalse(calls)
            resolver.fetch(item, retry_unavailable=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len([r for r in ledger.records() if r['kind'] == 'FULLTEXT_FETCHED']), 2)

    def test_failed_local_and_pmc_fall_back_to_verified_pdf(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 72), 'Ankle outcomes\n10.1234/ankle\nMethods: adults.')
            pdf = doc.tobytes()
        requests = []
        def respond(request):
            requests.append(str(request.url))
            return httpx.Response(404) if request.url.path.endswith('fullTextXML') else httpx.Response(200, content=pdf)
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            item = card().model_copy(update={'fulltext_required': True, 'doi': '10.1234/ankle',
                'fulltext_url': 'https://www.boa.ac.uk/article.pdf'})
            resolver = FullTextResolver(client=httpx.Client(transport=httpx.MockTransport(respond)), ledger=ledger,
                local_papers={item.evidence_id: {'path': str(Path(temp) / 'missing.pdf'),
                                               'sha256': '0' * 64, 'identity_confirmed': True}})
            result = resolver.fetch(item)
            self.assertEqual(result.access_level, 'FULL_TEXT')
            self.assertEqual(len(requests), 2)
            attempts = [r['payload'] for r in ledger.records() if r['kind'] == 'FULLTEXT_SOURCE_ATTEMPT']
            self.assertEqual([a['source'] for a in attempts], ['LOCAL', 'PMC', 'OFFICIAL_PDF'])
            self.assertEqual([a['status'] for a in attempts], ['UNAVAILABLE', 'UNAVAILABLE', 'ACQUIRED'])

    def test_only_explicit_retry_replaces_unavailable_cache(self):
        xml = '<article><front><article-meta><article-id pub-id-type="pmc">123</article-id></article-meta></front><body><sec>Methods: adults.</sec></body></article>'
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(503) if len(requests) == 1 else httpx.Response(200, text=xml)
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            item = card().model_copy(update={'fulltext_required': True})
            resolver = FullTextResolver(client=httpx.Client(transport=httpx.MockTransport(respond)), ledger=ledger)
            failed = resolver.fetch(item)
            self.assertTrue(failed.fulltext_status.startswith('UNAVAILABLE'))
            resolver.fetch(item)
            self.assertEqual(len(requests), 1)
            recovered = resolver.fetch(item, retry_unavailable=True)
            self.assertEqual(recovered.access_level, 'FULL_TEXT')
            changed = item.model_copy(update={'decision_issue_ids': ('new-issue',), 'priority_score': 99})
            cached = resolver.fetch(changed, retry_unavailable=True)
            self.assertEqual(cached.decision_issue_ids, ('new-issue',))
            self.assertEqual(cached.priority_score, 99)
            self.assertEqual(len(requests), 2)
            results = [r for r in ledger.records() if r['kind'] == 'FULLTEXT_FETCHED']
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0]['payload']['card']['fulltext_status'].startswith('UNAVAILABLE'))
