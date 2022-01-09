import itertools
from string import ascii_lowercase
from typing import Dict, Iterable, Iterator, Tuple


Point = Tuple[int, int]


def _iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


class GameConfig:
    def __init__(self, dimensions: Tuple[int, int], ship_lengths: Iterable[int]):
        self.dimensions = dimensions
        self.rows, self.cols = dimensions
        self.ships = self._identify_ships(ship_lengths)
        self.points = list(self._points(self.rows, self.cols))

    @classmethod
    def _identify_ships(cls, ship_lengths: Iterable[int]) -> Dict[str, int]:
        """Associate an ID (string) with each ship and return the associations as a dict."""
        ship_map = {
            ship_id: ship_length
            for (ship_id, ship_length) in zip(
                _iter_all_strings(), sorted(ship_lengths, reverse=True)
            )
        }
        return ship_map

    def __repr__(self):
        return f"<Game(dimensions=({self.rows}, {self.cols}), ships={self.ships})>"

    @staticmethod
    def _points(rows, cols) -> Iterator[Point]:
        for row in range(rows):
            for col in range(cols):
                yield (row, col)
