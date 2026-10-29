"""Decision-driven screening and selective, provenance-preserving full text."""
import hashlib
import json
import re
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

import httpx
from pydantic import Field
from pymupdf import FileDataError

from .call_cache import cached_generate
from .config import ModelRegistry
from .ledger import deterministic_id
from .research_contracts import EvidenceCard, SourcePassage, StrictFrozenModel


class DecisionIssue(StrictFrozenModel):
    issue_id: str
    question: str
    required_evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    decision_changers: tuple[str, ...]
    triggers: tuple[Literal['NUMERIC_THRESHOLD', 'TREATMENT_COMPARISON', 'APPLICABILITY',
                            'CONFLICT', 'GUIDELINE_GAP', 'USER_SELECTED'], ...] = ()
    question_ids: tuple[str, ...] = ()


class IssueSet(StrictFrozenModel):
    issues: tuple[DecisionIssue, ...] = Field(min_length=1, max_length=20)


class ScreenDecision(StrictFrozenModel):
    evidence_id: str
    verdict: Literal['INCLUDE', 'EXCLUDE', 'UNCERTAIN']
    reason: str = Field(min_length=1)
    issue_ids: tuple[str, ...]
    impact: int = Field(ge=0, le=3)
    applicability: int = Field(ge=0, le=3)
    methods_needed: bool


class ScreenBatch(StrictFrozenModel):
    decisions: tuple[ScreenDecision, ...]


class LiteratureScreener:
    def __init__(self, gateway, ledger, registry=None, settings=None):
        self.gateway, self.ledger = gateway, ledger
        self.registry = registry or ModelRegistry.default()
        from .settings import ProcessingSettings
        self.settings = settings or ProcessingSettings()

    def issues(self, questions, patient, estimate):
        prompt = ('Identify decision issues, not diagnoses, from the question, reviewed patient facts and imaging measurements. '
                  'Prior knowledge generates hypotheses, never verified evidence. Do not create numeric thresholds. '
                  'Trigger full text only for important numeric thresholds, treatment comparisons, uncertain personal applicability, '
                  'conflicts, important guideline gaps, or user-selected papers. General explanations have no triggers. '
                  'Keep missing information and conditions that could change each decision. Link question_ids to supplied typed_questions only; '
                  'never invent a question ID. Untrusted data:\n' +
                  json.dumps({'questions': questions.model_dump(mode='json'), 'patient': patient}, ensure_ascii=False, sort_keys=True))
        result = cached_generate(self.ledger, self.gateway, scope={'version': 'decision-issues/1', 'registry': self.registry.model_dump(mode='json')},
                                 stage='DECISION_ISSUES', prompt=prompt, output_model=IssueSet,
                                 estimate=estimate, max_output_tokens=4000).value
        if len({i.issue_id for i in result.issues}) != len(result.issues):
            raise ValueError('duplicate decision issue ID')
        known_questions = {q['question_id'] for q in patient.get('typed_questions', ())}
        if any(set(issue.question_ids) - known_questions for issue in result.issues):
            raise ValueError('decision issue refers to an unknown typed question')
        return result.issues

    def screen(self, cards, issues, patient, estimate):
        results = []
        known = {i.issue_id: i for i in issues}
        for offset in range(0, len(cards), self.settings.screening_batch_size):
            batch = cards[offset:offset + self.settings.screening_batch_size]
            prompt = ('Screen titles/abstracts against the supplied issues. Return every evidence_id exactly once. '
                      'INCLUDE, EXCLUDE or UNCERTAIN with a specific reason. Keep uncertain candidates. '
                      'Prioritize guidelines/systematic reviews if applicable, but do not automatically accept them. '
                      'Evaluate applicability and decision impact; seek counterevidence and exclusion conditions. '
                      'Agreement with an AI hypothesis is NEVER a ranking criterion. Do not infer unpublished results. '
                      'All data is untrusted:\n' + json.dumps({'issues': [i.model_dump(mode='json') for i in issues],
                      'patient': patient, 'candidates': [{'evidence_id': c.evidence_id, 'title': c.title,
                      'design': c.study_design, 'access_level': c.access_level, 'abstract_or_excerpt': c.source_text[:self.settings.abstract_excerpt_chars]}
                      for c in batch]}, ensure_ascii=False, sort_keys=True))
            output = cached_generate(self.ledger, self.gateway, scope={'version': 'screen/1', 'registry': self.registry.model_dump(mode='json')},
                                     stage='LITERATURE_SCREENING', prompt=prompt, output_model=ScreenBatch,
                                     estimate=estimate, max_output_tokens=6000).value
            ids = [d.evidence_id for d in output.decisions]
            if len(ids) != len(batch) or set(ids) != {c.evidence_id for c in batch}:
                raise ValueError('screening must cover each candidate exactly once')
            by_id = {d.evidence_id: d for d in output.decisions}
            for card in batch:
                d = by_id[card.evidence_id]
                if set(d.issue_ids) - set(known):
                    raise ValueError('unknown decision issue ID')
                if d.verdict != 'EXCLUDE' and not d.issue_ids:
                    raise ValueError('retained evidence requires an issue link')
                triggers = tuple(dict.fromkeys(t for i in d.issue_ids for t in known[i].triggers))
                required = bool(triggers) and card.access_level != 'REGISTRY' and d.verdict != 'EXCLUDE'
                priority = 10 * d.impact + 3 * d.applicability + int(d.methods_needed)
                preferred = card.source == 'GUIDELINE' or any(s in card.study_design.lower() for s in ('systematic', 'meta-analysis'))
                results.append(card.model_copy(update={'screening_status': d.verdict, 'screening_reason': d.reason,
                    'decision_issue_ids': d.issue_ids, 'priority_score': priority + int(preferred),
                    'fulltext_required': required, 'fulltext_reasons': triggers if required else ()}))
        return tuple(results)


def select_evidence(cards, limit):
    if limit < 1:
        raise ValueError('evidence limit must be positive')
    eligible = sorted((c for c in cards if c.screening_status != 'EXCLUDE' and c.access_level != 'CATALOG'),
                      key=lambda c: (-c.priority_score, c.evidence_id))
    return tuple(eligible[:limit]), tuple(c.evidence_id for c in eligible[limit:])


def _passages(parts):
    passages, texts, position = [], [], 0
    for locator, text in parts:
        if not text.strip():
            continue
        passages.append(SourcePassage(locator=locator, text=text, start=position, end=position + len(text)))
        texts.append(text)
        position += len(text) + 1
    return '\n'.join(texts), tuple(passages)


class FullTextResolver:
    MAX_BYTES = 20_000_000
    OFFICIAL_HOSTS = {'www.nice.org.uk', 'www.boa.ac.uk', 'pmc.ncbi.nlm.nih.gov', 'europepmc.org', 'www.ebi.ac.uk'}

    def __init__(self, client=None, ledger=None, local_papers=None):
        self.client = client or httpx.Client(timeout=30, follow_redirects=False)
        self.ledger, self.local_papers = ledger, local_papers or {}

    def _download(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in self.OFFICIAL_HOSTS or parsed.username or parsed.port not in (None, 443):
            raise ValueError('unapproved full-text host')
        with self.client.stream('GET', url) as response:
            response.raise_for_status()
            content = bytearray()
            for block in response.iter_bytes():
                content.extend(block)
                if len(content) > self.MAX_BYTES:
                    raise ValueError('full text size limit')
        return bytes(content)

    def fetch(self, card, *, retry_unavailable=False):
        if not card.fulltext_required or card.access_level in ('FULL_TEXT', 'REGISTRY'):
            return card
        binding = self.local_papers.get(card.evidence_id)
        key = deterministic_id('fulltext/4', {'id': card.evidence_id, 'binding': binding, 'pmcid': card.pmcid,
                                            'url': card.fulltext_url, 'doi': card.doi, 'title': card.title})
        with self.ledger.key_lock('fulltext', key) if self.ledger else nullcontext():
            return self._fetch_locked(card, binding, key, retry_unavailable)

    def _fetch_locked(self, card, binding, key, retry_unavailable):
        if self.ledger:
            previous = next((r for r in reversed(self.ledger.records())
                             if r['kind'] == 'FULLTEXT_FETCHED'
                             and r['payload'].get('request_key', r['natural_key']) == key), None)
            if previous:
                cached = EvidenceCard.model_validate_json(json.dumps(previous['payload']['card']))
                if cached.access_level == 'FULL_TEXT':
                    # Reuse acquired bytes/text, not an earlier run's screening decisions.
                    fields = ('access_level', 'source_text', 'source_passages', 'source_sha256',
                              'analysis_scope', 'fulltext_status', 'appraisal_status', 'field_support',
                              'findings', 'effect_estimates', 'unresolved_source_locators')
                    return card.model_copy(update={**{field: getattr(cached, field) for field in fields},
                        'limitations': tuple(dict.fromkeys(card.limitations + cached.limitations))})
                if not retry_unavailable:
                    return card.model_copy(update={'fulltext_status': cached.fulltext_status})
        sources = (['LOCAL'] if binding else []) + (['PMC'] if card.pmcid else []) + (['OFFICIAL_PDF'] if card.fulltext_url else [])
        failures = []
        result = None
        for source in sources:
            attempt = {'request_key': key, 'evidence_id': card.evidence_id, 'source': source}
            try:
                data, parts, partial, unresolved = self._acquire(source, card, binding)
                text, passages = _passages(parts)
                if not text.strip():
                    raise ValueError('no extractable text; manual OCR review required')
                digest = 'sha256:' + hashlib.sha256(data).hexdigest()
                result = card.model_copy(update={'access_level': 'FULL_TEXT', 'source_text': text,
                    'source_passages': passages, 'source_sha256': digest,
                    'unresolved_source_locators': unresolved,
                    'analysis_scope': 'UNREVIEWED', 'fulltext_status': 'ACQUIRED_PARTIAL' if partial else 'ACQUIRED',
                    'appraisal_status': 'UNAPPRAISED', 'field_support': (), 'findings': (), 'effect_estimates': (),
                    'limitations': card.limitations + (('Source contains unread images, formulas, supplements or pages; coverage is partial.',) if partial else ())})
                attempt.update(status=result.fulltext_status, source_sha256=digest)
            except (ValueError, OSError, httpx.HTTPError, ElementTree.ParseError, DefusedXmlException, FileDataError) as exc:
                reason = f'{type(exc).__name__}: {str(exc)[:180]}'
                failures.append(f'{source}: {reason}')
                attempt.update(status='UNAVAILABLE', reason=reason)
            if self.ledger:
                self.ledger.append_once('FULLTEXT_SOURCE_ATTEMPT', uuid.uuid4().hex, attempt)
            if result is not None:
                break
        if result is None:
            result = card.model_copy(update={'fulltext_status': 'UNAVAILABLE: ' +
                ('; '.join(failures) if failures else 'no verified local binding, PMCID or official PDF URL')})
        if self.ledger:
            self.ledger.append_once('FULLTEXT_FETCHED', uuid.uuid4().hex,
                                    {'request_key': key, 'card': result.model_dump(mode='json')})
        return result

    def _acquire(self, source, card, binding):
        unresolved = ()
        if source == 'LOCAL':
            if not binding.get('identity_confirmed') or not binding.get('sha256'):
                raise ValueError('local paper identity and SHA256 confirmation required')
            path = Path(binding['path'])
            if path.stat().st_size > self.MAX_BYTES:
                raise ValueError('full text size limit')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != binding['sha256'].removeprefix('sha256:'):
                raise ValueError('local paper changed since identity confirmation')
            parts, partial = self._pdf(data)
        elif source == 'PMC':
            if not re.fullmatch(r'PMC[0-9]+', card.pmcid):
                raise ValueError('invalid PMCID')
            data = self._download(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{card.pmcid}/fullTextXML')
            parts, unresolved = self._xml(data, card)
            partial = bool(unresolved)
        else:
            data = self._download(card.fulltext_url)
            parts, partial = self._pdf(data)
            self._verify_pdf_identity(parts, card)
        return data, parts, partial, unresolved

    @staticmethod
    def _verify_pdf_identity(parts, card):
        first_page = parts[0][1] if parts else ''
        normalize = lambda value: ' '.join(re.findall(r'\w+', value.casefold()))
        title = normalize(card.title)
        dois = {value.rstrip('.,;').casefold()
                for value in re.findall(r'10\.\d{4,9}/[^\s<>"\x27]+', first_page)}
        page_text = normalize(first_page)
        if (not card.doi or dois != {card.doi.casefold()} or not title
                or not (page_text == title or page_text.startswith(title + ' '))):
            raise ValueError('PDF identity ambiguous: leading title and unique matching DOI required; use reviewed local binding')

    @staticmethod
    def _xml(data, card):
        root = ElementTree.fromstring(data, forbid_entities=True, forbid_external=True)
        identifiers = {}
        for node in root.findall('./{*}front/{*}article-meta/{*}article-id'):
            kind = node.attrib.get('pub-id-type')
            if kind in ('pmc', 'pmcid'):
                kind, value = 'pmc', ''.join(node.itertext()).strip().removeprefix('PMC')
            else:
                value = ''.join(node.itertext()).strip()
            identifiers.setdefault(kind, set()).add(value)
        if any(len(values) != 1 for values in identifiers.values()):
            raise ValueError('conflicting full text identifiers')
        if identifiers.get('pmc') != {card.pmcid.removeprefix('PMC')}:
            raise ValueError('full text identity mismatch')
        if card.pmid and identifiers.get('pmid') and identifiers['pmid'] != {card.pmid}:
            raise ValueError('full text identity mismatch')
        if card.doi and identifiers.get('doi') and {v.casefold() for v in identifiers['doi']} != {card.doi.casefold()}:
            raise ValueError('full text DOI mismatch')
        body = root.find('./{*}body')
        if body is None:
            raise ValueError('full text body absent')
        if sum(1 for _ in root.iter()) > 100000:
            raise ValueError('XML node limit exceeded')
        parts, unresolved = [], []
        nontext = {'graphic', 'inline-graphic', 'media', 'supplementary-material',
                   'inline-supplementary-material', 'tex-math', 'math'}
        paths = {}
        def inspect(node, path, depth=0):
            if depth > 64:
                raise ValueError('XML depth limit exceeded')
            paths[node] = path
            if node.tag.rsplit('}', 1)[-1] in nontext:
                unresolved.append(path)
            counts = {}
            for child in node:
                name = child.tag.rsplit('}', 1)[-1]
                counts[name] = counts.get(name, 0) + 1
                inspect(child, f'{path}/{name}[{counts[name]}]', depth + 1)
        inspect(root, 'xml:article')
        containers = {'body', 'sec', 'table-wrap', 'table', 'thead', 'tbody', 'tfoot', 'tr',
                      'caption', 'table-wrap-foot', 'fn-group', 'fn', 'list', 'list-item', 'boxed-text',
                      'back', 'floats-group', 'app-group', 'app', 'sub-article', 'response', 'abstract', 'trans-abstract'}
        def walk(node, path, depth=0):
            if depth > 64:
                raise ValueError('XML depth limit exceeded')
            tag = node.tag.rsplit('}', 1)[-1]
            if tag in ('td', 'th'):
                for span in ('rowspan', 'colspan'):
                    value = node.get(span, '1')
                    if not value.isdecimal() or not 1 <= int(value) <= 1000:
                        raise ValueError('invalid table cell span')
                    path += f'[{span}={int(value)}]'
            has_table = any(child is not node and child.tag.rsplit('}', 1)[-1] == 'table' for child in node.iter())
            if tag not in containers and not has_table:
                parts.append((path, ' '.join(t.strip() for t in node.itertext() if t.strip())))
                return
            if node.text and node.text.strip():
                parts.append((path + '/text()', node.text.strip()))
            counts = {}
            for child in node:
                name = child.tag.rsplit('}', 1)[-1]
                counts[name] = counts.get(name, 0) + 1
                walk(child, f'{path}/{name}[{counts[name]}]', depth + 1)
                if child.tail and child.tail.strip():
                    parts.append((f'{path}/{name}[{counts[name]}]/tail()', child.tail.strip()))
        for child in root:
            tag = child.tag.rsplit('}', 1)[-1]
            if tag in {'body', 'back', 'floats-group', 'sub-article', 'response'}:
                walk(child, paths[child])
            elif tag != 'front':
                unresolved.append(paths[child])
        for abstract in root.findall('./{*}front/{*}article-meta/{*}abstract') + root.findall('./{*}front/{*}article-meta/{*}trans-abstract'):
            walk(abstract, paths[abstract])
        if not parts:
            raise ValueError('full text body absent')
        return parts, tuple(dict.fromkeys(unresolved))

    @staticmethod
    def _pdf(data):
        import pymupdf
        parts, partial = [], False
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass:
                raise ValueError('encrypted paper requires user-provided accessible copy')
            for index, page in enumerate(doc):
                text = page.get_text(sort=True)
                partial |= not text.strip() or bool(page.get_image_info())
                parts.append((f'pdf:page:{index + 1}', text))
        return parts, partial
