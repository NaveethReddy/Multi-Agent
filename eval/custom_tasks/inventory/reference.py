import json


class Inventory:
    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def add_item(self, name: str, qty: int, price: float) -> None:
        if qty < 0:
            raise ValueError("qty must be >= 0")
        if name in self._items:
            self._items[name]["qty"] += qty
        else:
            self._items[name] = {"qty": qty, "price": price}

    def remove_stock(self, name: str, qty: int) -> None:
        item = self._items[name]
        if qty > item["qty"]:
            raise ValueError("insufficient stock")
        item["qty"] -= qty

    def total_value(self) -> float:
        return sum(i["qty"] * i["price"] for i in self._items.values())

    def low_stock(self, threshold: int) -> list:
        return sorted(n for n, i in self._items.items() if i["qty"] < threshold)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._items, f)

    @classmethod
    def load(cls, path: str) -> "Inventory":
        inv = cls()
        with open(path) as f:
            inv._items = json.load(f)
        return inv
