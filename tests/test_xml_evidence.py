import unittest
import tempfile
from pathlib import Path
import httpx

from medic.literature import FullTextResolver
from tests.test_decision_literature import card


FRONT = '<front><article-meta><article-id pub-id-type="pmc">123</article-id></article-meta></front>'


class XmlEvidenceTests(unittest.TestCase):
    def test_appendices_and_floating_tables_are_not_omitted(self):
        xml = '<article>' + FRONT + '<body><p>Methods.</p></body><floats-group><table-wrap><table><tr><td>42.8</td></tr></table></table-wrap></floats-group><back><app-group><app><p>Excluded severe deformity.</p></app></app-group></back></article>'
        result = self.fetch(xml.encode())
        self.assertIn('42.8', result.source_text)
        self.assertIn('Excluded severe deformity.', result.source_text)
        self.assertTrue(any('floats-group' in p.locator for p in result.source_passages))

    def test_unread_graphics_keep_fulltext_coverage_partial(self):
        xml = '<article>' + FRONT + '<body><sec><p>Results <inline-graphic href="image.png"/></p><table-wrap><caption><p>Outcomes</p></caption><graphic href="table.png"/></table-wrap></sec></body></article>'
        result = self.fetch(xml.encode())
        self.assertEqual(result.fulltext_status, 'ACQUIRED_PARTIAL')
        self.assertEqual(len(result.unresolved_source_locators), 2)
        self.assertTrue(all('graphic' in p for p in result.unresolved_source_locators))
        from medic.evidence_appraisal import EvidenceAppraiser
        from medic.ledger import AppendOnlyJsonlLedger
        from medic.providers import ApprovedGateway, RecordedGateway
        from medic.config import ModelRegistry
        with tempfile.TemporaryDirectory() as temp:
            response = {'fields': [], 'applicability': [], 'limitations': [], 'missing_fields': ['findings']}
            appraiser = EvidenceAppraiser(ApprovedGateway(RecordedGateway('anthropic', 'model', [response])),
                AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl'))
            analyzed = appraiser.appraise(result, {}, ModelRegistry.default().estimate_tribunal_cost(100).approve())
            self.assertEqual(analyzed.analysis_scope, 'PARTIAL_FULL_TEXT')
            self.assertEqual(analyzed.unresolved_source_locators, result.unresolved_source_locators)

    def test_repeated_table_value_requires_unambiguous_location(self):
        from medic.evidence_appraisal import EvidenceAppraiser
        from medic.research_contracts import SourcePassage
        from medic.ledger import AppendOnlyJsonlLedger
        from medic.providers import ApprovedGateway, RecordedGateway
        from medic.config import ModelRegistry
        item = card().model_copy(update={'access_level': 'FULL_TEXT', 'fulltext_status': 'ACQUIRED',
            'source_text': '12.5\n12.5', 'source_sha256': 'sha256:test', 'source_passages': (
                SourcePassage(locator='xml:body/table/tr[1]/td[1]', text='12.5', start=0, end=4),
                SourcePassage(locator='xml:body/table/tr[2]/td[1]', text='12.5', start=5, end=9))})
        for locator in (None, 'xml:body/table/tr[2]/td[1]', 'invented'):
            with self.subTest(locator=locator), tempfile.TemporaryDirectory() as temp:
                field = {'field': 'effect_estimates', 'value': '12.5', 'quote': '12.5', 'source_locator': locator}
                response = {'fields': [field], 'applicability': [], 'limitations': [], 'missing_fields': []}
                appraiser = EvidenceAppraiser(ApprovedGateway(RecordedGateway('anthropic', 'model', [response])),
                    AppendOnlyJsonlLedger(Path(temp) / 'ledger.jsonl'))
                estimate = ModelRegistry.default().estimate_tribunal_cost(100).approve()
                if locator is None or locator == 'invented':
                    with self.assertRaises(ValueError):
                        appraiser.appraise(item, {}, estimate)
                else:
                    result = appraiser.appraise(item, {}, estimate)
                    self.assertEqual(result.field_support[0].source_start, 5)
                    self.assertEqual(result.field_support[0].source_locator, locator)

    def fetch(self, xml):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=xml)))
        return FullTextResolver(client=client).fetch(card().model_copy(update={'fulltext_required': True}))

    def test_utf16_entity_declarations_are_rejected(self):
        xml = '<!DOCTYPE article [<!ENTITY unsafe "invented result">]><article>' + FRONT + '<body><sec><p>&unsafe;</p></sec></body></article>'
        result = self.fetch(xml.encode('utf-16'))
        self.assertEqual(result.access_level, 'ABSTRACT')
        self.assertTrue(result.fulltext_status.startswith('UNAVAILABLE'))

    def test_conflicting_article_identifiers_are_rejected(self):
        front = FRONT.replace('</article-meta>', '<article-id pub-id-type="pmc">456</article-id><article-id pub-id-type="pmc">123</article-id></article-meta>')
        result = self.fetch(('<article>' + front + '<body><p>Results.</p></body></article>').encode())
        self.assertEqual(result.access_level, 'ABSTRACT')

    def test_table_cells_have_unique_source_paths_and_original_spans(self):
        xml = '<article>' + FRONT + '''<body><sec><title>Results</title><p>Observed outcomes.</p>
          <table-wrap id="T1"><label>Table 1</label><caption><p>Follow-up</p></caption><table>
          <thead><tr><th>Group</th><th colspan="2">Outcome</th></tr></thead>
          <tbody><tr><td rowspan="2">Control</td><td>12.5</td><td>mm</td></tr>
          <tr><td>8.1</td><td>mm</td></tr></tbody></table>
          <table-wrap-foot><p>Values are means.</p></table-wrap-foot></table-wrap></sec></body></article>'''
        result = self.fetch(xml.encode())
        self.assertEqual(result.access_level, 'FULL_TEXT')
        cells = [p for p in result.source_passages if '/td[' in p.locator or '/th[' in p.locator]
        self.assertEqual(len(cells), 7)
        self.assertEqual(len({p.locator for p in cells}), 7)
        self.assertTrue(any('rowspan=2' in p.locator for p in cells))
        self.assertTrue(any('colspan=2' in p.locator for p in cells))
        self.assertEqual(result.source_text.count('12.5'), 1)
        self.assertIn('Values are means.', result.source_text)
        for passage in result.source_passages:
            self.assertEqual(result.source_text[passage.start:passage.end], passage.text)

    def test_default_namespace_article_remains_readable(self):
        result = self.fetch(('<article xmlns="urn:jats">' + FRONT + '<body><p>Results.</p></body></article>').encode())
        self.assertEqual(result.access_level, 'FULL_TEXT')
