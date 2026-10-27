import re
import time
import httpx


def validate_query(query, identifiers=()):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError('invalid search query')
    patterns = (r'[\w.+-]+@[\w.-]+', r'[A-Za-z]:[\\/]', r'\\\\', r'\b\d{6}[- ]?\d{7}\b',
                r'\b\d{2,3}[- ]\d{3,4}[- ]\d{4}\b', r'https?://', r'[\x00-\x1f]')
    if any(re.search(p, query) for p in patterns) or any(str(v).casefold() in query.casefold() for v in identifiers if v):
        raise ValueError('possible identifying information in external search query')
    return query.strip()


def get(client, url, *, params=None):
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                delay = response.headers.get('Retry-After', '')
                time.sleep(min(float(delay), 10) if delay.isdigit() else 2 ** attempt)
                continue
            return response.raise_for_status()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
