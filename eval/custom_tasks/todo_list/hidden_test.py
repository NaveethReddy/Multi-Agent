import pytest
from todo import TodoList

def test_ids_increase():
    t = TodoList(); assert t.add("a") == 1 and t.add("b") == 2

def test_pending_and_complete():
    t = TodoList(); a = t.add("a"); t.add("b"); t.add("c")
    t.complete(a)
    assert t.pending() == ["b", "c"]

def test_remove():
    t = TodoList(); a = t.add("a"); t.remove(a)
    assert t.pending() == []
    with pytest.raises(KeyError): t.remove(a)

def test_unknown_complete():
    with pytest.raises(KeyError): TodoList().complete(3)

def test_empty_text():
    with pytest.raises(ValueError): TodoList().add("   ")

def test_ids_not_reused_after_remove():
    t = TodoList(); a = t.add("a"); t.remove(a)
    assert t.add("b") == 2
