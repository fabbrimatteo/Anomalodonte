from typing import Callable
from typing import Generic
from typing import TypeVar


BufferElement = TypeVar('BufferElement')


class SortedBuffer(Generic[BufferElement]):

    def __init__(self, buffer_size, sort_key=None):
        # type: (int, Callable[[BufferElement], float]) -> None
        """
        Buffer that only keeps the top `buffer_size` elements
        (elements with the highest value), with ranking defined
        by the `sort_key` function.

        :param buffer_size: maximum size of the buffer
        :param sort_key: function used for ranking elements;
            >> default: identity function
        """
        self.size = buffer_size

        if sort_key is None:
            self.key = lambda x: x
        else:
            self.key = sort_key

        self.buffer = []


    def append(self, x):
        # type: (BufferElement) -> None
        """
        Append new element x to the sorted buffer, than sort the buffer
        again, and remove the lowest scoring item if the buffer has
        exceeded the maximum size.

        :param x: new element to add
        """
        self.buffer.append(x)
        self.buffer.sort(key=self.key, reverse=True)
        if len(self.buffer) > self.size:
            self.buffer.pop()


    def __getitem__(self, i):
        # type: (int) -> BufferElement
        return self.buffer[i]


    def __str__(self):
        # type: () -> str
        return f'{self.buffer}'


def demo():
    import random

    buffer = SortedBuffer[int](buffer_size=4)
    for i in range(10):
        x = random.randint(0, 100)
        buffer.append(x)
        print(f'({i}): adding {x} to buffer -> {buffer}')


if __name__ == '__main__':
    demo()
