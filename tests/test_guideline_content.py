import unittest
import httpx

from medic.evidence import OfficialAnkleGuidelineSource


class GuidelineContentTests(unittest.TestCase):
    def test_fetches_official_content_without_inventing_revision_date(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, text='<html><body><main><h1>Guidance</h1><p>Recommendation body.</p></main></body></html>',
                                  headers={"content-type": "text/html", "last-modified": "observed-server-revision"})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            cards = OfficialAnkleGuidelineSource(client=client).search("ankle fracture")
        self.assertEqual(len(requests), 2)
        self.assertTrue(all("Recommendation body." in card.source_text for card in cards))
        self.assertTrue(all(card.appraisal_status == "UNAPPRAISED" for card in cards))
        self.assertFalse(any("2026 ankle osteoarthritis entry" in card.study_design for card in cards))
