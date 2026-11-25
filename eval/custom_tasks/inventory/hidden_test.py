import json
import pytest
from inventory import Inventory

def test_add_merge_and_value():
    i = Inventory(); i.add_item("pen", 10, 2.0); i.add_item("pen", 5, 99.0)
    assert i.total_value() == pytest.approx(30.0)

def test_remove_stock():
    i = Inventory(); i.add_item("pen", 3, 1.0); i.remove_stock("pen", 2)
    with pytest.raises(ValueError): i.remove_stock("pen", 5)
    with pytest.raises(KeyError): i.remove_stock("cup", 1)

def test_low_stock_sorted():
    i = Inventory(); i.add_item("b", 1, 1); i.add_item("a", 2, 1); i.add_item("c", 50, 1)
    assert i.low_stock(5) == ["a", "b"]

def test_negative_qty():
    with pytest.raises(ValueError): Inventory().add_item("x", -1, 1.0)

def test_save_load(tmp_path):
    i = Inventory(); i.add_item("pen", 4, 2.5)
    p = tmp_path / "inv.json"; i.save(str(p))
    assert json.loads(p.read_text())
    j = Inventory.load(str(p))
    assert j.total_value() == pytest.approx(10.0)
