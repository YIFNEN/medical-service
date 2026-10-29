import pytest


class Node:
    def __init__(self, **attributes):
        self.attributes = attributes

    def GetAttribute(self, name):
        return self.attributes.get(name)

    def SetAttribute(self, name, value):
        self.attributes[name] = value


def test_markups_require_explicit_matching_volume_binding():
    from scripts.slicer_review import bind_markup, validate_markup_binding
    volume = Node(**{'Medic.SeriesUID': 'selected', 'Medic.OriginalHashes': '["hash"]'})
    node = Node()
    with pytest.raises(ValueError, match='bound'):
        validate_markup_binding(volume, node)
    bind_markup(volume, node)
    validate_markup_binding(volume, node)
    foreign = Node(**{'Medic.SeriesUID': 'foreign', 'Medic.OriginalHashes': '["hash"]'})
    with pytest.raises(ValueError, match='bound'):
        bind_markup(volume, foreign)
    with pytest.raises(ValueError, match='bound'):
        validate_markup_binding(volume, foreign)
