import random
from abc import ABC, abstractmethod
from queue import Queue
from typing import Generator, Iterable, List, Set, Tuple

from .board import Board, Orientation, Point, format_board, initialize_board
from .enemy import Enemy
from .game import GameConfig
from .tools import enumerate_all_points


DEBUG = False


# -----------------------------------------------------------------------------


class BattleShipStrategy(ABC):
    @abstractmethod
    def __init__(self, game_config: GameConfig, enemy: Enemy):
        self.game = game_config
        self.enemy = enemy

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


class ProbabilityStrategy(BattleShipStrategy):
    pass


def _points_until_blocked(board: Board, points: Iterable[Point]) -> int:
    available = 0
    for (r, c) in points:
        if not board[r][c]:
            available += 1
        else:
            break
    return available


def _possible_placements(board: Board, point: Point, ship_length: int) -> int:
    """The number of ways a ship of given length may be placed over a point.

    The number of possible placements is equal to the sum of the number of possible
    vertical and horizontal placements.
    """
    assert 0 <= point[0] < board.rows, "Row value out-of-bounds"
    assert 0 <= point[1] < board.cols, "Col value out-of-bounds"

    row, col = point

    # Vertical
    ## Up
    points = [
        (r, col) for r in range(row - 1, row - ship_length, -1) if 0 <= r < board.rows
    ]
    available_up = _points_until_blocked(board, points)

    ## Down
    points = [
        (r, col) for r in range(row + 1, row + ship_length) if 0 <= r < board.rows
    ]
    available_down = _points_until_blocked(board, points)

    ## Vertical total
    available_vertical = available_up + available_down
    placements_vertical = max(available_vertical - ship_length + 2, 0)

    # Horizontal
    ## Left
    points = [
        (row, c) for c in range(col - 1, col - ship_length, -1) if 0 <= c < board.cols
    ]
    available_left = _points_until_blocked(board, points)

    ## Right
    points = [
        (row, c) for c in range(col + 1, col + ship_length) if 0 <= c < board.cols
    ]
    available_right = _points_until_blocked(board, points)

    ## Horizontal total
    available_horizontal = available_left + available_right
    placements_horizontal = max(available_horizontal - ship_length + 2, 0)

    return placements_vertical + placements_horizontal


def _subtract_possible_placements(
    game_board: Board, placements_board: Board, point: Point, ship_length: int
):
    """TODO: Give a description"""
    assert game_board.rows == placements_board.rows, "Unequal row length"
    assert game_board.cols == placements_board.cols, "Unequal column length"
    assert 0 <= point[0] < game_board.rows, "Row value out-of-bounds"
    assert 0 <= point[1] < game_board.cols, "Col value out-of-bounds"

    row, col = point

    # Indicate that the given point now has no "possibilities".
    placements_board[row][col] = 0


def _initial_ship_odds(game: GameConfig) -> Board:
    board = Board(game.rows, game.cols)

    for row in range(game.rows):
        for col in range(game.cols):
            for ship in game.ships.values():
                point = (row, col)
                board[row][col] += _possible_placements(board, point, ship)

    return board


# -----------------------------------------------------------------------------
# Logic for sampling the board / sinking ships


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
    enemy: Enemy,
):
    ship_map = {ship.id: ship for ship in enemy._ships}

    # is_hit, is_sunk, ship_id = _apply_missile(point, ships)
    is_hit, is_sunk, ship_id = enemy.apply_missile(point)
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
            # is_hit, is_sunk, border_ship_id = _apply_missile(border_point, ships)
            is_hit, is_sunk, border_ship_id = enemy.apply_missile(border_point)

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
                    # is_hit, is_sunk, ship_id_next = _apply_missile(p, ships)
                    is_hit, is_sunk, ship_id_next = enemy.apply_missile(p)
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
                        # is_hit, is_sunk, ship_id_next = _apply_missile(p, ships)
                        is_hit, is_sunk, ship_id_next = enemy.apply_missile(p)
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
    points = set(enumerate_all_points(rows, cols))
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
    points_remaining = set(enumerate_all_points(rows, cols))
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
    def __init__(self, game_config: GameConfig, enemy: Enemy):
        super().__init__(game_config, enemy)

        self._is_solved = False
        self._finished = False
        self._turns = []
        self._search_strategy = random_checkerboard_search
        self._hit_strategy = hit_strategy_full_context
        self._points_remaining = set(
            enumerate_all_points(self.game.rows, self.game.cols)
        )

    def step(self) -> Point:
        return (0, 0)

    def solve(self):
        if self._finished:
            return (self._is_solved, self._turns, self._points_remaining)

        _ship_lengths = tuple(length for length in self.game.ships.values())
        points_to_search = self._search_strategy(
            self.game.rows, self.game.cols, _ship_lengths
        )

        # Loop while any ships remain
        while not self.enemy.all_sunk():
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
                self.game.dimensions, point, self._points_remaining, self.enemy
            )

            self._turns.extend(additional_points)
            self._points_remaining -= set(additional_points)

        self._finished = True
        self._is_solved = self.enemy.all_sunk()

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
