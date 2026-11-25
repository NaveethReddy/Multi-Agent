class TodoList:
    def __init__(self) -> None:
        self._tasks: dict[int, list] = {}
        self._next = 1

    def add(self, text: str) -> int:
        if not text.strip():
            raise ValueError("empty text")
        tid = self._next
        self._next += 1
        self._tasks[tid] = [text, False]
        return tid

    def complete(self, task_id: int) -> None:
        self._tasks[task_id][1] = True

    def remove(self, task_id: int) -> None:
        del self._tasks[task_id]

    def pending(self) -> list:
        return [t for t, done in self._tasks.values() if not done]
