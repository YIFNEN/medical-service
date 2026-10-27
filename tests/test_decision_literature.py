import tempfile
import unittest
from pathlib import Path
from datetime import UTC, datetime
import httpx

from medic.config import ModelRegistry
from medic.evidence import normalize_europe_pmc
from medic.ledger import AppendOnlyJsonlLedger
from medic.providers import ApprovedGateway, RecordedGateway


def card():
    return normalize_europe_pmc({'id': 'PMC123', 'pmcid': 'PMC123', 'pmid': '12', 'title': 'Ankle outcomes',
                                'abstractText': 'Adults were studied.'}, datetime.now(UTC))


class DecisionLiteratureTests(unittest.TestCase):
    def test_official_pdf_requires_matching_first_page_identity(self):
        import pymupdf
        from medic.literature import FullTextResolver
        item = card().model_copy(update={'pmcid': None, 'doi': '10.1234/ankle',
            'fulltext_required': True, 'fulltext_url': 'https://www.boa.ac.uk/paper.pdf'})
        for title, doi, expected in (
            ('Unrelated paper', '10.1234/ankle', 'ABSTRACT'),
            ('Ankle outcomes', '10.1234/ankle-other', 'ABSTRACT'),
            ('Unrelated article\n10.9999/unrelated\nReferences:\nAnkle outcomes', '10.1234/ankle', 'ABSTRACT'),
            ('Ankle outcomes\n10.9999/unrelated', '10.1234/ankle', 'ABSTRACT'),
            ('Ankle outcomes', '10.1234/ankle', 'FULL_TEXT'),
        ):
            with self.subTest(title=title, doi=doi):
                with pymupdf.open() as doc:
                    doc.new_page().insert_text((72, 72), title + '\n' + doi + '\nMethods: adults.')
                    data = doc.tobytes()
                client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=data)))
                result = FullTextResolver(client=client).fetch(item)
                self.assertEqual(result.access_level, expected)

    def test_appraisal_tracks_chunk_offsets_and_partial_coverage(self):
        from medic.evidence_appraisal import EvidenceAppraiser
        from medic.research_contracts import SourcePassage
        text = 'A' * 20000 + 'Twenty adults.'
        payload = {'fields': [{'field': 'population', 'value': '20 adults', 'quote': 'Twenty adults.'}],
                   'applicability': [], 'limitations': [], 'missing_fields': []}
        blank = {'fields': [], 'applicability': [], 'limitations': [], 'missing_fields': ['population']}
        item = card().model_copy(update={'access_level': 'FULL_TEXT', 'fulltext_status': 'ACQUIRED',
            'source_text': text, 'source_sha256': 'sha256:test',
            'source_passages': (SourcePassage(locator='pdf:page:2', text='Twenty adults.', start=20000, end=len(text)),)})
        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl')
            appraiser = EvidenceAppraiser(ApprovedGateway(RecordedGateway('anthropic', 'model', [blank, payload])), ledger)
            result = appraiser.appraise(item, {}, ModelRegistry.default().estimate_tribunal_cost(100).approve())
            self.assertEqual(result.field_support[0].source_start, 20000)
            self.assertEqual(result.field_support[0].source_locator, 'pdf:page:2')
            self.assertEqual(result.analysis_scope, 'FULL_TEXT')

    def test_screening_retains_uncertain_and_records_fulltext_trigger(self):
        from medic.literature import DecisionIssue, LiteratureScreener
        issue = DecisionIssue(issue_id='i', question='Threshold?', required_evidence=('measurement method',),
                              missing_information=(), decision_changers=('different threshold',),
                              triggers=('NUMERIC_THRESHOLD',))
        item = card()
        response = {'decisions': [{'evidence_id': item.evidence_id, 'verdict': 'UNCERTAIN', 'reason': 'Need methods',
                                  'issue_ids': ['i'], 'impact': 3, 'applicability': 2, 'methods_needed': True}]}
        with tempfile.TemporaryDirectory() as temp:
            raw = RecordedGateway('anthropic', 'model', [response])
            service = LiteratureScreener(ApprovedGateway(raw), AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl'))
            result = service.screen((item,), (issue,), {}, ModelRegistry.default().estimate_tribunal_cost(100).approve())
            self.assertEqual(result[0].screening_status, 'UNCERTAIN')
            self.assertTrue(result[0].fulltext_required)
            self.assertEqual(service.screen((item,), (issue,), {}, ModelRegistry.default().estimate_tribunal_cost(100).approve()), result)
            self.assertEqual(raw.call_count, 1)

    def test_general_question_does_not_fetch_fulltext(self):
        from medic.literature import FullTextResolver
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(500)
        result = FullTextResolver(client=httpx.Client(transport=httpx.MockTransport(respond))).fetch(card())
        self.assertEqual(result.access_level, 'ABSTRACT')
        self.assertFalse(calls)

    def test_fulltext_xml_identity_and_sections_are_verified(self):
        from medic.literature import FullTextResolver
        xml = '<article><front><article-meta><article-id pub-id-type="pmc">123</article-id></article-meta></front><body><sec><title>Methods</title><p>Twenty adults.</p></sec><sec><title>Results</title><table-wrap><table><tr><td>12 months</td></tr></table></table-wrap></sec></body></article>'
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=xml)))
        result = FullTextResolver(client=client).fetch(card().model_copy(update={'fulltext_required': True}))
        self.assertEqual(result.access_level, 'FULL_TEXT')
        self.assertIn('12 months', result.source_text)
        self.assertTrue(result.source_passages)
        self.assertEqual(result.analysis_scope, 'UNREVIEWED')
        wrong = xml.replace('>123<', '>456<')
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=wrong)))
        result = FullTextResolver(client=client).fetch(card().model_copy(update={'fulltext_required': True}))
        self.assertEqual(result.access_level, 'ABSTRACT')
        self.assertIn('identity', result.fulltext_status.lower())

    def test_priority_does_not_depend_on_agreement_and_limit_is_pending(self):
        from medic.literature import select_evidence
        one = card().model_copy(update={'screening_status': 'INCLUDE', 'priority_score': 1})
        two = one.model_copy(update={'evidence_id': 'second', 'priority_score': 8})
        selected, pending = select_evidence((one, two), 1)
        self.assertEqual(selected[0].evidence_id, 'second')
        self.assertEqual(pending, (one.evidence_id,))
