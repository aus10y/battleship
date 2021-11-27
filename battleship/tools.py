from typing import Iterable


def enumerate_all_points(rows: int, cols: int) -> Iterable:
    for row in range(rows):
        for col in range(cols):
            yield (row, col)
