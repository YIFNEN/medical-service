from datetime import UTC, datetime
from html.parser import HTMLParser

import httpx

from .ledger import deterministic_id
from .research_contracts import EvidenceCard


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skipped = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skipped += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skipped = max(0, self.skipped - 1)

    def handle_data(self, data):
        if not self.skipped and data.strip():
            self.parts.append(data.strip())


class OfficialAnkleGuidelineSource:
    """Fetch selected official pages; do not claim comprehensive guideline search."""
    ENTRIES = (
        ("NICE NG38 recommendations", "https://www.nice.org.uk/guidance/ng38/chapter/recommendations"),
        ("BOASt management of ankle fractures", "https://www.boa.ac.uk/resource/boast-12-pdf.html"),
    )
    query_sensitive = False

    def __init__(self, client=None):
        self.client = client or httpx.Client(timeout=30, follow_redirects=True)

    def search(self, query):
        cards = []
        for title, url in self.ENTRIES:
            response = self.client.get(url)
            response.raise_for_status()
            parser = _PageText()
            parser.feed(response.text)
            text = "\n".join(parser.parts)
            cards.append(EvidenceCard(
                evidence_id=deterministic_id("evidence", {"source": "GUIDELINE", "url": url}),
                source="GUIDELINE", title=title, url=str(response.url), retrieved_at=datetime.now(UTC),
                study_design="official guideline webpage; revision currency not independently established",
                population="requires content extraction", findings=(), source_text=text,
                access_level="WEBPAGE", appraisal_status="UNAPPRAISED" if text else "NO_CONTENT",
                limitations=("Accessed webpage only; linked PDFs were not read. Verify recommendation scope and revision before application.",
                             "Selected official sources; not a comprehensive guideline search.",
                             "HTTP Last-Modified: " + response.headers.get("last-modified", "not supplied")),
            ))
        return tuple(cards)
