import pytest
from students import StudentManager

def test_add_and_search():
    m = StudentManager(); m.add(2, "Bob", 80.0); m.add(1, "alice", 90.5)
    assert [r["id"] for r in m.search("")] == [1, 2]
    assert m.search("ALI")[0]["name"] == "alice"

def test_duplicate_id():
    m = StudentManager(); m.add(1, "A", 1.0)
    with pytest.raises(ValueError): m.add(1, "B", 2.0)

def test_delete():
    m = StudentManager(); m.add(1, "A", 1.0); m.delete(1)
    assert m.search("A") == []
    with pytest.raises(KeyError): m.delete(1)

def test_update_partial():
    m = StudentManager(); m.add(1, "A", 50.0)
    m.update(1, grade=75.0)
    r = m.search("A")[0]
    assert r["grade"] == 75.0 and r["name"] == "A"
    m.update(1, name="Zed")
    assert m.search("zed")[0]["grade"] == 75.0
    with pytest.raises(KeyError): m.update(9, name="x")

def test_record_shape():
    m = StudentManager(); m.add(5, "Eve", 88.0)
    assert m.search("eve") == [{"id": 5, "name": "Eve", "grade": 88.0}]
