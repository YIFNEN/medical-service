"""Publication notices and conservative study counts; no inferred independence."""


def publication_status(updates, publication_types=()):
    relations = {u.get('relation') for u in updates}
    if 'RetractionIn' in relations or any(str(t).casefold() == 'retracted publication' for t in publication_types):
        return 'RETRACTED'
    if 'ExpressionOfConcernIn' in relations:
        return 'EXPRESSION_OF_CONCERN'
    if relations & {'ErratumIn', 'CorrectedandRepublishedIn', 'RetractedandRepublishedIn', 'UpdateIn'}:
        return 'CORRECTION_REQUIRES_REVIEW'
    return 'SOURCE_CHECKED_NO_NOTICE'


def study_counts(cards):
    reviews = [c for c in cards if any(term in c.study_design.casefold() for term in ('systematic', 'meta-analysis', 'guideline'))
               or c.source == 'GUIDELINE']
    primary = [c for c in cards if c not in reviews and c.access_level != 'REGISTRY']
    mapped = [c for c in primary if c.study_id and c.study_link_status == 'USER_REVIEWED']
    study_ids = {c.study_id for c in mapped}
    included = {sid for c in reviews if c.study_link_status == 'USER_REVIEWED' for sid in c.included_study_ids}
    return {'selected_cards': len(cards), 'verified_unique_primary_studies': len(study_ids),
            'unmapped_primary_cards': len(primary) - len(mapped), 'review_cards': len(reviews),
            'registry_cards': sum(c.access_level == 'REGISTRY' for c in cards),
            'review_primary_overlaps': sorted(study_ids & included),
            'independence_status': 'PARTIALLY_MAPPED_NOT_A_TOTAL_INDEPENDENT_STUDY_COUNT',
            'formal_risk_of_bias': 'NOT_PERFORMED', 'GRADE': 'NOT_PERFORMED', 'meta_analysis': 'NOT_PERFORMED'}
