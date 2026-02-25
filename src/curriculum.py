from typing import List

class LengthCurriculum:
    """
    iterate examples from short -> long prompts
    """
    def __init__(self, items: List[tuple[str, str]]):
        self.items = sorted(items, key=lambda x: len(x[0]))
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self.items):
            raise StopIteration
        it = self.items[self._i]
        self._i += 1
        return it
    
