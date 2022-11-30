如何写一个 python 迭代器

```python
from collections import deque

class MyRange(object):
    def __init__(self, end):
        self.start = 0
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end:
            ret = self.start
            self.start += 1
            return ret
        else:
            raise StopIteration

    def run(self):
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        try:
            self.is_run = True
            deque(
                self,
                maxlen=0)  # feed the entire iterator into a zero-length deque
            info = 'hello'
        finally:
            self.is_run = False

        return info


if __name__ == '__main__':
    a = MyRange(5)
    print(next(a))
    print(next(a))
    print(next(a))
    print(next(a))
    print(next(a))
    print(a.run())
    print(next(a))
```
