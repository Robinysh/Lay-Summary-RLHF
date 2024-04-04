import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def time_it(name=None) -> Iterator[None]:
    yield
    return
    # pylint: disable=unreachable
    print(f"Begin {name}")
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"Finish {name}: {1000*(toc - tic):.3f}ms")
