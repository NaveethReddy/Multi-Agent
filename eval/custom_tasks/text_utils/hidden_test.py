import pytest
from textutils import word_frequency, most_common, is_anagram

def test_freq():
    assert word_frequency("The cat. The dog!") == {"the": 2, "cat": 1, "dog": 1}
    assert word_frequency("") == {}

def test_most_common_ties():
    assert most_common("b a b a c", 2) == [("a", 2), ("b", 2)]
    assert most_common("x y", 5) == [("x", 1), ("y", 1)]

def test_negative_n():
    with pytest.raises(ValueError): most_common("a", -1)

def test_anagram():
    assert is_anagram("Dormitory", "dirty room") is True
    assert is_anagram("abc", "abd") is False
