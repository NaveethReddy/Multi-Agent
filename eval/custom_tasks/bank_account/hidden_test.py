import pytest
from bank import Account

def test_deposit_withdraw():
    a = Account("x", 10); assert a.deposit(5) == 15; assert a.withdraw(3) == 12; assert a.balance == 12

def test_invalid_amounts():
    a = Account("x", 10)
    for bad in (0, -1):
        with pytest.raises(ValueError): a.deposit(bad)
        with pytest.raises(ValueError): a.withdraw(bad)

def test_overdraft():
    a = Account("x", 10)
    with pytest.raises(ValueError): a.withdraw(11)
    assert a.balance == 10

def test_transfer_ok_and_atomic():
    a, b = Account("a", 50), Account("b", 0)
    a.transfer(b, 20); assert (a.balance, b.balance) == (30, 20)
    with pytest.raises(ValueError): a.transfer(b, 100)
    assert (a.balance, b.balance) == (30, 20)

def test_balance_read_only():
    a = Account("x", 1)
    with pytest.raises(AttributeError): a.balance = 5
