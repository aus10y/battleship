from abc import ABC, abstractmethod
import code
from collections import namedtuple
import itertools
import random

from enum import Enum
from pprint import pprint, pformat
from queue import Queue
from string import ascii_lowercase
from timeit import default_timer as timer
from typing import Callable, Dict, Generator, Iterable, Set, Tuple, List, Type, Union


DEBUG = False


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
        True

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


# -----------------------------------------------------------------------------


class Game:
    def __init__(self, rows, cols, ships):
        self.rows = rows
        self.cols = cols
        self.ships = ships


class BattleShipStrategy(ABC):
    @abstractmethod
    def __init__(self, dimensions: Tuple[int, int], ships: Tuple[Ship, ...]):
        self.dimensions = dimensions
        self._rows, self._cols = dimensions
        self.ships = ships

    @abstractmethod
    def step(self) -> Point:
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def is_solved(self) -> bool:
        pass

    @abstractmethod
    def solution(self):
        pass


# -----------------------------------------------------------------------------
# Logic for sampling the board / sinking ships


def _apply_missile(
    point: Point, ships: Tuple[Ship, ...]
) -> Tuple[bool, bool, Union[str, None]]:
    is_hit = False
    is_sunk = False
    ship_id = None

    for ship in ships:
        # if point not in placement.points:
        #    continue
        if ship.is_hit(point):
            return (True, ship.is_sunk(), ship.id)

    return (is_hit, is_sunk, ship_id)


def _ships_remain(ships: Tuple[Ship, ...]) -> bool:
    # return any(p.points for p in ship_placements)
    return any(not ship.is_sunk() for ship in ships)


def simulate_game(
    dimensions: Tuple[int, int],
    ship_sizes: Tuple[int, ...],
    StrategyClass: Type[BattleShipStrategy],
):
    rows, cols = dimensions

    ships = generate_ship_placements(rows, cols, ship_sizes)
    pprint(ships)
    # board = initialize_board(dimensions)
    # set_board(board, ship_placements)

    strategy = StrategyClass(dimensions, ships)
    turns = strategy.solve()
    return (strategy.is_solved(), turns)


def run_simulation(
    dimensions: Tuple[int, int],
    ships: Tuple[Ship, ...],
    search_strategy: Callable,
    hit_strategy: Callable,
) -> Tuple[bool, List[Point], Set[Point]]:
    rows, cols = dimensions

    turns = []
    points_remaining = set(_enumerate_all_points(rows, cols))

    _ship_lengths = tuple(s.length for s in ships)
    points_to_search = search_strategy(rows, cols, _ship_lengths)

    # Loop while any ships remain
    while _ships_remain(ships):
        # Pick a point to test
        try:
            point = next(points_to_search)
        except StopIteration:
            break

        if point not in points_remaining:
            if DEBUG:
                print(f"- Point '{point}' already tested")
            continue

        # Apply hit strategy
        additional_points = hit_strategy(dimensions, point, points_remaining, ships)

        turns.extend(additional_points)
        points_remaining -= set(additional_points)

    return (not _ships_remain(ships), turns, points_remaining)


def _get_border_points(point: Point) -> Tuple[Tuple[Orientation, Point], ...]:
    row, col = point
    return (
        (Orientation.Right, (row, col + 1)),
        (Orientation.Down, (row + 1, col)),
        (Orientation.Left, (row, col - 1)),
        (Orientation.Up, (row - 1, col)),
    )


def _points_along_vector(point: Point, other: Point, length: int) -> List[Point]:
    row, col = point
    other_row, other_col = other

    points = []

    # Guard against the same point being passed for both args.
    if point == other:
        return []

    # Assume that the points are neighbors and share a border (but never an edge).
    row_change = row != other_row
    if row_change:
        delta = other_row - row
        points = [(other_row + (delta * i), col) for i in range(1, length - 1)]
    else:
        delta = other_col - col
        points = [(row, other_col + (delta * i)) for i in range(1, length - 1)]

    return points


def hit_strategy_full_context(
    dimensions: Tuple[int, int],
    point: Point,
    points_available: Set[Point],
    ships: Tuple[Ship, ...],
):
    ship_map = {ship.id: ship for ship in ships}

    is_hit, is_sunk, ship_id = _apply_missile(point, ships)
    points_tested = [point]
    points_available.remove(point)

    if not is_hit:
        return points_tested

    if is_sunk:
        if DEBUG:
            if DEBUG:
                print(f"-- Point '{point}': SANK '{ship_id}'")
        return points_tested

    if DEBUG:
        print(f"-- Point '{point}': HIT '{ship_id}'")

    queue = Queue()
    queue.put((point, ship_id))

    while not queue.empty():
        sank_ship = False
        change_direction = False

        ship_point, ship_id = queue.get_nowait()
        ship = ship_map[ship_id]

        # Generate the four bordering points.
        border_points = _get_border_points(ship_point)

        # Keep only the bordering points that are in the set of available points.
        border_points = [
            (orientation, p)
            for (orientation, p) in border_points
            if p in points_available
        ]

        if DEBUG:
            print(f">> Point '{ship_point}', border points: '{border_points}'")

        # Determine the direction along which the ship lies.
        # The points are iterated over at random (since 'border_points')
        # is a Set).
        for _, border_point in border_points:

            # Test the point.
            is_hit, is_sunk, border_ship_id = _apply_missile(border_point, ships)

            points_tested.append(border_point)
            points_available.remove(border_point)

            # If the point was not a hit, try another of the bordering points.
            if not is_hit:
                if DEBUG:
                    print(f"--- Border Point '{border_point}': MISS")
                continue

            if border_ship_id != ship_id:
                # The point was a hit, but on on the ship of interest.
                # Push the hit ship to the queue for later (if not sunk).
                if is_sunk:
                    if DEBUG:
                        print(
                            f"--- Border Point '{border_point}': SANK '{border_ship_id}"
                        )
                    break
                else:
                    queue.put_nowait((border_point, border_ship_id))
                    continue
            else:
                # We've hit another point of our current ship.

                # If sunk, then we can move on.
                if is_sunk:
                    if DEBUG:
                        print(f"--- Border Point '{border_point}': SANK '{ship_id}")
                    break
                else:
                    if DEBUG:
                        print(
                            f"--- Border Point '{border_point}': HIT '{border_ship_id}'"
                        )
                    pass

                points_ahead = []
                for p in _points_along_vector(ship_point, border_point, ship.length):
                    if p not in points_available:
                        break
                    points_ahead.append(p)

                # - Continue testing points in this direction until the ship is
                #   sunk or the current ship is missed.
                if DEBUG:
                    print(f">> Points ahead ('{border_point}'): {points_ahead}")
                for p in points_ahead:
                    is_hit, is_sunk, ship_id_next = _apply_missile(p, ships)
                    points_tested.append(p)
                    points_available.remove(p)

                    if is_sunk:
                        if DEBUG:
                            print(f"---- Point '{p}': SANK '{ship_id}'")
                        sank_ship = True
                        break

                    if not is_hit:
                        if DEBUG:
                            print(f"---- Point '{p}': miss, change direction")
                        change_direction = True
                        break
                    else:
                        if ship_id_next != ship_id:
                            if DEBUG:
                                print(
                                    f"---- Point '{p}': HIT other ship {ship_id_next}"
                                )
                            queue.put_nowait((p, ship_id_next))
                            break
                        else:
                            if DEBUG:
                                print(
                                    f"---- Point '{p}': HIT '{ship_id}', continue along vector"
                                )
                            pass

                if change_direction:
                    points_behind = []
                    for p in _points_along_vector(
                        border_point,
                        ship_point,
                        ship.length,
                    ):
                        if p not in points_available:
                            break
                        points_behind.append(p)

                    # - Continue testing points in this direction until the ship is
                    #   sunk or the current ship is missed.
                    if DEBUG:
                        print(f">> Points behind ('{border_point}'): {points_behind}")
                    for p in points_behind:
                        is_hit, is_sunk, ship_id_next = _apply_missile(p, ships)
                        points_tested.append(p)
                        points_available.remove(p)

                        if is_sunk:
                            if DEBUG:
                                print(f"---- Point '{p}': SANK '{ship_id}'")
                            sank_ship = True
                            break

                        if not is_hit:
                            # Should not occur
                            if DEBUG:
                                print(f"---- !!!! Point '{p}': miss")
                            break
                        else:
                            if ship_id_next != ship_id:
                                print("?????")
                                queue.put_nowait((p, ship_id_next))
                                break
                            else:
                                if DEBUG:
                                    print(f"---- Point '{p}': HIT '{ship_id}'")
                                pass

                if sank_ship:
                    break

    return points_tested


def random_search(rows: int, cols: int, ships: Tuple[int, ...]) -> Generator:
    """
    True random search.
    """
    points = set(_enumerate_all_points(rows, cols))
    yield from points


def random_checkerboard_search(
    rows: int, cols: int, ships: Tuple[int, ...]
) -> Generator:
    """
    Random search from shuffled minimum covering grid.
    """
    min_ship_size = min(ships)

    shifts = list(range(min_ship_size))
    random.shuffle(shifts)

    points_to_search = set()

    for starting_row, shift in enumerate(shifts):
        for row in range(starting_row, rows, min_ship_size):
            for col in range(shift, cols, min_ship_size):
                point = (row, col)
                points_to_search.add(point)

    yield from points_to_search


def random_checkerboard_priority_search(
    rows: int, cols: int, ships: Tuple[int, ...]
) -> Generator:

    # All points
    points_remaining = set(_enumerate_all_points(rows, cols))
    points_tested = set()

    # Produce set of points for largest ship
    max_ship_size = max(ships)

    shifts = list(range(max_ship_size))
    random.shuffle(shifts)

    search_points = set()

    for starting_row, shift in enumerate(shifts):
        for row in range(starting_row, rows, max_ship_size):
            for col in range(shift, cols, max_ship_size):
                point = (row, col)
                search_points.add(point)

    yield from search_points

    points_remaining = points_remaining - search_points
    points_tested.update(search_points)

    # ...
    pass


def random_checkerboard_quadrant_search(
    rows: int, cols: int, ships: Tuple[int, ...]
) -> Generator:
    points_to_search = set(random_checkerboard_search(rows, cols, ships))

    quadrants = (set(), set(), set(), set())
    for point in points_to_search:
        row, col = point
        q_index = int((row // (rows / 2)) + (2 * (col // (cols / 2))))
        quadrants[q_index].add(point)

    while any(quadrants):
        for quadrant in quadrants:
            if not quadrant:
                continue
            point = quadrant.pop()
            points_to_search.remove(point)
            yield point


class CheckerboardSparseFirst(BattleShipStrategy):
    def __init__(self, dimensions: Tuple[int, int], ships: Tuple[Ship, ...]):
        super().__init__(dimensions, ships)

        self._is_solved = False
        self._finished = False
        self._turns = []
        self._search_strategy = random_checkerboard_search
        self._hit_strategy = hit_strategy_full_context
        self._points_remaining = set(_enumerate_all_points(self._rows, self._cols))

    def step(self) -> Point:
        return (0, 0)

    def solve(self):
        if self._finished:
            return (self._is_solved, self._turns, self._points_remaining)

        _ship_lengths = tuple(s.length for s in self.ships)
        points_to_search = self._search_strategy(self._rows, self._cols, _ship_lengths)

        # Loop while any ships remain
        while _ships_remain(self.ships):
            # Pick a point to test
            try:
                point = next(points_to_search)
            except StopIteration:
                break

            if point not in self._points_remaining:
                if DEBUG:
                    print(f"- Point '{point}' already tested")
                continue

            # Apply hit strategy
            additional_points = self._hit_strategy(
                self.dimensions, point, self._points_remaining, self.ships
            )

            self._turns.extend(additional_points)
            self._points_remaining -= set(additional_points)

        self._finished = True
        self._is_solved = not _ships_remain(self.ships)

        return (self._is_solved, self._turns, self._points_remaining)

    def is_solved(self) -> bool:
        return self._is_solved

    def solution(self):
        return self._turns


def evaluate_search(rows: int, cols: int, ships: Tuple[int, ...], search_strategy):
    search_points = search_strategy(rows, cols, ships)
    board = initialize_board((rows, cols))

    print(format_board(board))
    print("\n--------------------\n")

    for point in search_points:
        row, col = point
        board[row][col] = "O"
        print(format_board(board))
        print("\n--------------------\n")


def play_one_game(rows, cols, ships):
    ship_placements = generate_ship_placements(rows, cols, ships)
    board = initialize_board((rows, cols))
    set_board(board, ship_placements)
    print(format_board(board))
    print()

    result = run_simulation(
        (rows, cols),
        ship_placements,
        random_checkerboard_search,
        hit_strategy_full_context,
    )

    print("--------------------")
    print(f"Sunk: {result[0]}")
    print(f"Turns: {len(result[1])}")
    print("--------------------")

    return (result, ship_placements)


def print_histogram(bins):
    for turns in sorted(bins):
        print(f"{turns},{bins[turns]}")


def analyze_results(bins: Dict[int, int], histogram=True):
    turns_taken = sum(turns * count for turns, count in bins.items())
    games_played = sum(count for _, count in bins.items())

    # Min & Max
    min_turn, max_turn = min(bins), max(bins)

    # Mean
    mean_turns = turns_taken / games_played

    # Median
    def _flatten_bins(binned_turns):
        for turns, counts in (
            (turns, binned_turns[turns]) for turns in sorted(binned_turns)
        ):
            for _ in range(counts):
                yield turns

    _start = games_played // 2
    _stop = (_start + 2) if games_played % 2 == 0 else (_start + 1)
    _midpoint = list(itertools.islice(_flatten_bins(bins), _start, _stop))

    median_turns = sum(_midpoint) / len(_midpoint)

    # -----

    print(f"- min turns: {min_turn}")
    print(f"- max turns: {max_turn}")
    print(f"- mean turns:   {mean_turns}")
    print(f"- median turns: {median_turns}")
    if histogram:
        print(f"- Histogram:")
        print_histogram(bins)


def run_simulations(rows, cols, ships, search_strategy, hit_strategy, runs=1000):
    bins = dict()

    start_time = timer()
    for _ in range(runs):
        ship_placements = generate_ship_placements(rows, cols, ships)
        board = initialize_board((rows, cols))
        set_board(board, ship_placements)

        sank_all, turns, remaining = run_simulation(
            (rows, cols),
            ship_placements,
            search_strategy,
            hit_strategy,
        )

        if not sank_all:
            print("!!")
            print(format_board(board))

        turn_count = len(turns)
        if turn_count not in bins:
            bins[turn_count] = 0
        bins[turn_count] += 1

    end_time = timer()

    print(f"Summary:")
    print(f"- Time: {end_time - start_time:0.2f}s")
    analyze_results(bins)

    return bins


if __name__ == "__main__":
    ROWS, COLS = BOARD_DIMENSIONS

    start = timer()
    ship_placements = generate_ship_placements(ROWS, COLS, SHIPS)
    stop = timer()

    board = initialize_board(BOARD_DIMENSIONS)
    set_board(board, ship_placements)

    print(f"\nShip Placements ({stop - start:0.6f}s): \n{pformat(ship_placements)}\n")
    print(f"\nBoard: \n{format_board(board)}\n")

    # -----

    boards = [initialize_board(BOARD_DIMENSIONS) for _ in range(100)]
    for board in boards:
        ship_placements = generate_ship_placements(ROWS, COLS, SHIPS)
        # set_board(board, ship_placements, _determine_ship_identity)

    # write_games("battleship_boards.txt", SHIPS, boards)
