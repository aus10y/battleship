from typing import Iterable, Set, Tuple, Union

from .board import Point, Ship


class Enemy:
    def __init__(self, dimensions: Tuple[int, int], ships: Iterable[Ship]):
        self._ships = tuple(ships)
        self._rows, self._cols = dimensions
        is_valid, self._points = self._validate(self._ships)
        if not is_valid:
            raise Exception("Invalid ship placement(s)")

    def __repr__(self):
        return "<Enemy(TODO)>"

    @classmethod
    def _validate(cls, ships: Iterable[Ship]) -> Tuple[bool, Set[Point]]:
        is_valid = True
        points = set()

        ship_points = (point for ship in ships for point in ship.points)
        for point in ship_points:
            if point in points:
                is_valid = False
                break
            else:
                points.add(point)

        return (is_valid, points)

    def apply_missile(self, point: Point) -> Tuple[bool, bool, Union[str, None]]:
        """Indicate whether the missile strikes and sinks a ship."""

        # Quick check.
        if point not in self._points:
            return (False, False, None)

        for ship in self._ships:
            if ship.is_hit(point):
                return (True, ship.is_sunk(), ship.id)

        return (False, False, None)

    def all_sunk(self) -> bool:
        return all(ship.is_sunk() for ship in self._ships)
