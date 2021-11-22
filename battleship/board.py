import itertools
import random

from collections import namedtuple
from enum import Enum
from string import ascii_lowercase
from typing import Callable, Dict, Generator, Iterable, Set, Tuple, List, Type, Union


"""
Convenient Type Aliases
"""
Board = List[List[str]]
Point = Tuple[int, int]


"""
Location (0, 0) will be defined as the top left corner of the board.
"""
BOARD_DIMENSIONS = (10, 10)  # (row, col)
# BOARD_DIMENSIONS = (20, 20)  # (row, col)


"""
Ships are treated as one-dimensional, having only a length.
"""
SHIPS = (2, 3, 3, 4, 5)
# SHIPS = (2, 3, 3, 4, 5, 5, 5, 7, 10, 12)


"""
Battleships can have four possible rotations.

Orientation.Up is defined as the ship body moving "up" from the chosen point
(the "X" below):

  | |
  | |
  | |
  | |
  |X|

The remaining Orientations are defined similarly.
"""


class Orientation(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Ship:
    def __init__(self, ship_id: str, points: Iterable[Point]):
        self.id = ship_id
        self._points = set(points)
        self._hits = set()
        self.length = len(self._points)

    def __str__(self):
        return str(sorted(self._points))

    def __repr__(self):
        points = ", ".join(str(p) for p in sorted(self._points))
        return f"<Ship({self.id}, {points})>"

    def is_valid(self) -> bool:
        return True

    def is_hit(self, point: Point) -> bool:
        if point in self._points:
            self._hits.add(point)
            return True
        return False

    def is_sunk(self) -> bool:
        return self._points == self._hits


def get_board_dimensions(board: Board):
    row = len(board)
    col = len(board[0]) if row else 0
    return (row, col)


def initialize_board(dimensions: Tuple[int, int]) -> Board:
    """
    Return an empty board.

    Missile placements are represented with a boolean, and are initially False.
    """
    row, col = dimensions
    return [["-" for _ in range(col)] for _ in range(row)]


def format_board(board: Board):
    return "\n".join("".join(row) for row in board)


def format_board_flat(board: Board):
    return ",".join("".join(row) for row in board)


def set_board(board: Board, ships: Tuple[Ship, ...]):
    for ship in ships:
        ship_size = ship.length
        for row, col in ship._points:
            board[row][col] = ship.id


def fill_board(board: Board, points: Set[Point]):
    for point in points:
        row, col = point
        board[row][col] = "O"


def write_games(file_name: str, ships: Tuple[int, ...], boards: Iterable[Board]):
    with open(file_name, "w") as f:
        f.write(",".join(str(ship) for ship in ships))
        f.write("\n")
        for board in boards:
            f.write(format_board_flat(board))
            f.write("\n")


def read_games(file_name: str) -> Tuple[Tuple[int], List[Board]]:
    # TODO
    ...


# -----------------------------------------------------------------------------
# Logic for placing ships


def _ship_points(point: Point, ship: int, orientation: Orientation) -> Set[Point]:
    row, col = point

    if orientation is Orientation.Up:
        transform = lambda r, c, i: (r - i, c)
    elif orientation is Orientation.Down:
        transform = lambda r, c, i: (r + i, c)
    elif orientation is Orientation.Left:
        transform = lambda r, c, i: (r, c - i)
    else:  # orientation is Orientation.Right
        transform = lambda r, c, i: (r, c + i)

    return set(transform(row, col, inc) for inc in range(ship))


def _points_of_ship_conflict(
    point: Point, ship: int, orientation: Orientation
) -> Set[Point]:
    row, col = point

    if orientation is Orientation.Up:
        transform = lambda r, c, i: (r + i, c)
    elif orientation is Orientation.Down:
        transform = lambda r, c, i: (r - i, c)
    elif orientation is Orientation.Left:
        transform = lambda r, c, i: (r, c + i)
    else:  # orientation is Orientation.Right
        transform = lambda r, c, i: (r, c - i)

    return set(transform(row, col, inc) for inc in range(ship))


def _points_of_boundary_conflict(
    rows: int, cols: int, ship: int, orientation: Orientation
) -> Set[Point]:
    row_from, row_to = 0, rows
    col_from, col_to = 0, cols

    if orientation is Orientation.Up:
        row_to = ship - 1
    elif orientation is Orientation.Down:
        row_from = rows - ship + 1
    elif orientation is Orientation.Left:
        col_to = ship - 1
    else:  # orientation is Orientation.Right
        col_from = cols - ship + 1

    # Produce the set of points from which the ship would overlap a boundary.
    points = set(
        (row, col) for row in range(row_from, row_to) for col in range(col_from, col_to)
    )

    return points


def _available_orientations(rows: int, cols: int, ship: int) -> List[Orientation]:
    orientations = []

    fits_row = ship <= rows
    fits_col = ship <= cols

    if fits_row:
        orientations.extend((Orientation.Left, Orientation.Right))
    if fits_col:
        orientations.extend((Orientation.Up, Orientation.Down))

    return orientations


def select_ship_placement(
    rows: int,
    cols: int,
    ship: int,
    points_available: Set[Point],
    points_unavailable: Set[Point],
) -> Set[Point]:
    """
    Choose a set of valid points for a given ship.
    """
    points_remaining = set(points_available)

    # Pick a random orientation.
    orientation = random.choice(_available_orientations(rows, cols, ship))

    # Remove the set of points that would cause the ship to overlap the board
    # boundaries.
    points_remaining -= _points_of_boundary_conflict(rows, cols, ship, orientation)

    # Remove the set of points that would cause the ship to overlap other ships
    # that have already been placed.
    conflict_sets = (
        _points_of_ship_conflict(p, ship, orientation) for p in points_unavailable
    )
    for conflict_set in conflict_sets:
        points_remaining -= conflict_set

    # Pick a random point from the set of remaining points.
    point = random.choice(tuple(points_remaining))

    # Rooted at the chosen point, find and return all points corresponding to
    # the ship & orientation.
    return _ship_points(point, ship, orientation)


def _enumerate_all_points(rows: int, cols: int) -> Iterable:
    for row in range(rows):
        for col in range(cols):
            yield (row, col)


def generate_ship_placements(
    rows: int, cols: int, ships: Tuple[int, ...]
) -> Tuple[Ship, ...]:
    """
    Generates a set of points for each ship.
    """
    points_available = set(_enumerate_all_points(rows, cols))
    points_unavailable = set()
    placements = []

    """
    There are some improvements that could be made to the placement strategy.
    - The order that the ships are placed could be randomized.
    - The order that the ships are placed could be from largest to smallest.
    - When placing ships (perhaps from largest to smallest), if a ship cannot
      fit on the board, back up and re-place the last ship in a different
      position.
    """

    for ship_id, ship in zip(iter_all_strings(), sorted(ships, reverse=True)):
        ship_points = select_ship_placement(
            rows, cols, ship, points_available, points_unavailable
        )
        placements.append(Ship(ship_id, ship_points))  # , len(ship_points)))
        points_available -= ship_points
        points_unavailable.update(ship_points)

    return tuple(placements)


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)
