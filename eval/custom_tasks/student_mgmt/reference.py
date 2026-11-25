"""Student management."""


class StudentManager:
    def __init__(self) -> None:
        self._s: dict[int, dict] = {}

    def add(self, student_id: int, name: str, grade: float) -> None:
        if student_id in self._s:
            raise ValueError(f"duplicate id {student_id}")
        self._s[student_id] = {"id": student_id, "name": name, "grade": grade}

    def delete(self, student_id: int) -> None:
        del self._s[student_id]

    def update(self, student_id: int, name=None, grade=None) -> None:
        rec = self._s[student_id]
        if name is not None:
            rec["name"] = name
        if grade is not None:
            rec["grade"] = grade

    def search(self, query: str) -> list:
        q = query.lower()
        return [dict(r) for _, r in sorted(self._s.items()) if q in r["name"].lower()]
