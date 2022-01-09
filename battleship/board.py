import random
from enum import Enum
from typing import Any, Iterable, Iterator, List, Set, Tuple

from .game import GameConfig, Point
from .tools import enumerate_all_points

"""
Convenient Type Aliases
"""
SimpleBoard = List[List[str]]


"""
Location (0, 0) will be defined as the top left corner of the board.
"""
BOARD_DIMENSIONS = (10, 10)  # (row, col)


"""
Ships are treated as one-dimensional, having only a length.
"""
SHIPS = (2, 3, 3, 4, 5)


class Ship:
    def __init__(self, ship_id: str, points: Iterable[Point]):
        self.id = ship_id
        self.points = set(points)
        self._hits = set()
        self.length = len(self.points)

    def __str__(self):
        return str(sorted(self.points))

    def __repr__(self):
        points = ", ".join(str(p) for p in sorted(self.points))
        return f"<Ship({self.id}, {points})>"

    def is_valid(self) -> bool:
        return True

    def is_hit(self, point: Point) -> bool:
        if point in self.points:
            self._hits.add(point)
            return True
        return False

    def is_sunk(self) -> bool:
        return self.points == self._hits


class Board:
    def __init__(self, rows: int, cols: int, initial_point=0):
        self.rows = rows
        self.cols = cols
        self._board = tuple([initial_point for _ in range(cols)] for _ in range(rows))

    def __getitem__(self, row):
        return self._board[row]

    @staticmethod
    def _row_fmt_helper(row_num, row) -> Iterable[str]:
        yield str(row_num)
        for value in row:
            yield str(value)

    def __repr__(self):
        # return "\n".join("".join(f"{col:>3}" for col in row) for row in self._board)
        row_width = len(str(self.rows))
        col_width = len(str(self.cols))

        header = "{}{}".format(
            " " * (row_width + 1),
            " ".join(
                f"{str(col_num).rjust(col_width)}" for col_num in range(self.cols)
            ),
        )

        body = "\n".join(
            " ".join(s.rjust(col_width) for s in self._row_fmt_helper(row_num, row))
            for row_num, row in enumerate(self._board)
        )

        return f"{header}\n{body}"

    def _points(self) -> Iterator[Point]:
        for row in range(self.rows):
            for col in range(self.cols):
                yield (row, col)

    def get(self, point: Point) -> Any:
        row, col = point
        return self[row][col]

    def set(self, point: Point, value: Any):
        row, col = point
        self[row][col] = value

    def points(self) -> Iterator[Point]:
        return self._points()

    def values(self) -> Iterator:
        return (self._board[row][col] for row, col in self._points())

    def items(self) -> Iterator:
        return (((row, col), self._board[row][col]) for row, col in self._points())


def get_board_dimensions(board: SimpleBoard):
    row = len(board)
    col = len(board[0]) if row else 0
    return (row, col)


def initialize_board(dimensions: Tuple[int, int]) -> SimpleBoard:
    """
    Return an empty board.

    Missile placements are represented with a boolean, and are initially False.
    """
    row, col = dimensions
    return [["-" for _ in range(col)] for _ in range(row)]


def format_board(board: SimpleBoard):
    return "\n".join("".join(row) for row in board)


def format_board_flat(board: SimpleBoard):
    return ",".join("".join(row) for row in board)


def set_board(board: SimpleBoard, ships: Tuple[Ship, ...]):
    for ship in ships:
        for row, col in ship.points:
            board[row][col] = ship.id


def fill_board(board: SimpleBoard, points: Set[Point]):
    for point in points:
        row, col = point
        board[row][col] = "O"


def write_games(file_name: str, ships: Tuple[int, ...], boards: Iterable[SimpleBoard]):
    with open(file_name, "w") as f:
        f.write(",".join(str(ship) for ship in ships))
        f.write("\n")
        for board in boards:
            f.write(format_board_flat(board))
            f.write("\n")


def read_games(file_name: str) -> Tuple[Tuple[int], List[SimpleBoard]]:
    # TODO
    ...


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


def generate_ship_placements(
    # rows: int, cols: int, ships: Tuple[int, ...]
    game_config: GameConfig,
) -> Tuple[Ship, ...]:
    """
    Generates a set of points for each ship.
    """
    points_available = set(enumerate_all_points(game_config.rows, game_config.cols))
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
    largest_ship_first = (
        (ship_id, game_config.ships[ship_id])
        for ship_id in sorted(
            game_config.ships, key=lambda k: game_config.ships[k], reverse=True
        )
    )

    for ship_id, ship_length in largest_ship_first:
        ship_points = select_ship_placement(
            game_config.rows,
            game_config.cols,
            ship_length,
            points_available,
            points_unavailable,
        )
        placements.append(Ship(ship_id, ship_points))  # , len(ship_points)))
        points_available -= ship_points
        points_unavailable.update(ship_points)

    return tuple(placements)


def select_ship_placement(
    rows: int,
    cols: int,
    ship_length: int,
    points_available: Set[Point],
    points_unavailable: Set[Point],
) -> Set[Point]:
    """
    Choose a set of valid points for a given ship.
    """
    points_remaining = set(points_available)

    # Pick a random orientation.
    orientation = random.choice(_available_orientations(rows, cols, ship_length))

    # Remove the set of points that would cause the ship to overlap the board
    # boundaries.
    points_remaining -= _points_of_boundary_conflict(
        rows, cols, ship_length, orientation
    )

    # Remove the set of points that would cause the ship to overlap other ships
    # that have already been placed.
    conflict_sets = (
        _points_of_ship_conflict(p, ship_length, orientation)
        for p in points_unavailable
    )
    for conflict_set in conflict_sets:
        points_remaining -= conflict_set

    # Pick a random point from the set of remaining points.
    point = random.choice(tuple(points_remaining))

    # Rooted at the chosen point, find and return all points corresponding to
    # the ship & orientation.
    return _ship_points(point, ship_length, orientation)
