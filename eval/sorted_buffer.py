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


def debug():
    b = SortedBuffer[int](buffer_size=4, sort_key=lambda x: -x)
    for i in range(10):
        b.append(i)
        print(f'{i}: {b.buffer}')


if __name__ == '__main__':
    debug()
