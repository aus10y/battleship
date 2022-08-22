from enum import Enum
from itertools import groupby, permutations
import random
from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict, Generator, Iterable, Iterator, List, Set, Tuple, Union

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

    @abstractmethod
    def turns(self) -> int:
        pass


# -----------------------------------------------------------------------------


class PointStatus(Enum):
    Unknown = None
    Miss = 1
    Hit = 2


class PointEstimate:
    __slots__ = ("permutations", "probability")

    def __init__(self, permutations: int, probability: float):
        self.permutations: int = permutations
        self.probability: float = probability

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"PointEstimate({self.permutations}, {self.probability:0.3f})"

    def set(self, permutations: int, probability: float):
        self.permutations = permutations
        self.probability = probability


"""
def _subtract_possible_placements(
    game_board: Board, placements_board: Board, point: Point, ship_length: int
):
    # TODO: Give a description
    assert game_board.rows == placements_board.rows, "Unequal row length"
    assert game_board.cols == placements_board.cols, "Unequal column length"
    assert 0 <= point[0] < game_board.rows, "Row value out-of-bounds"
    assert 0 <= point[1] < game_board.cols, "Col value out-of-bounds"

    row, col = point

    # Indicate that the given point now has no "possibilities".
    placements_board[row][col] = 0
"""


class ProbabilitySearch:
    # For each ship:
    # - Maintain a set of "candidate" points.
    #   - Each point could have the current possibilities paired with it.
    # - At the beginning of the game, each ship will have all points as candidates.

    def __init__(self, game: GameConfig):
        self.game = game
        self._point_map: Dict[str, Dict[Point, int]] = self._initial_candidates(game)

        self._ship_hits: Dict[str, Set[Point]] = {
            ship_id: set() for ship_id in self.game.ships
        }

        self._points_deduced: Set[Point] = set()

        self._hits: Set[Point] = set()
        self._misses: Set[Point] = set()

        self._ranked_points = []
        self._update_point_ranking()

    @property
    def _points_tested(self):
        return self._hits | self._misses

    @property
    def probability_board(self) -> Board:
        board = Board(self.game.rows, self.game.cols, initial_point=0.0)
        point_cache = {p: prob for p, _, prob in self._ranked_points}
        for point in board.points():
            probability = point_cache.get(point, 0.0)
            if probability:
                board.set(point, f"{probability:0.3f}")
            else:
                board.set(point, "     ")
        return board

    @property
    def permutation_board(self) -> Board:
        board = Board(self.game.rows, self.game.cols, initial_point=0)
        for point in board.points():
            permutations = 0
            for ship in self.game.ships:
                if point in self._point_map[ship]:
                    permutations += self._point_map[ship][point]
            board.set(point, permutations)
        return board

    def hit(self, point_hit: Point, ship_hit: str, verbose=False):
        self._hits.add(point_hit)
        self._ship_hits[ship_hit].add(point_hit)

        for ship, ship_length in self.game.ships.items():
            if ship == ship_hit:
                self._evaluate_hit(ship, ship_length, point_hit)
            else:
                self._evaluate_miss(ship, ship_length, point_hit)

        self._progate_certainties()
        self._update_point_ranking()

    def miss(self, point_missed: Point, verbose=False):
        # When a ship is missed, the estimated possible permutations and
        # probabilities need to be reduced in the are around the point,
        # for each remaining ship.

        self._misses.add(point_missed)

        # Produce a set of candidate points that may need to be re-scored.
        for ship, ship_length in self.game.ships.items():
            self._evaluate_miss(ship, ship_length, point_missed)

        while True:
            if not self._progate_certainties():
                break
        self._update_point_ranking()

    def sunk(self, ship_id: str):
        # TODO: Remove all remaining candidates when a ship is sunk.
        pass

    def _evaluate_hit(self, ship: str, ship_length: int, point: Point):
        candidates = set(
            point
            for point in self._points_available_all(
                point,
                self.game.dimensions,
                ship_length,
                set(self._point_map[ship]) | self._ship_hits[ship],
            )
            if point in self._point_map[ship]
        )

        self._point_map[ship].clear()

        for candidate in candidates:
            # When calculating the placements, the candidates should include
            # points already hit for the current ship.
            permutations = self._possible_placements_2(
                self.game,
                candidate,
                ship,
                set(candidates) | self._ship_hits[ship],
            )
            if permutations:
                self._point_map[ship][candidate] = permutations

    def _evaluate_miss(self, ship: str, ship_length: int, point: Point):
        # If the ship has no permutations, it must (should) already be sunk.
        if not self._point_map[ship]:
            return

        # If the point was not a candidate for this ship, we shouldn't need
        # to re-calculate the permutations & probabilities for it's
        # neighboring points.
        if point not in self._point_map[ship]:
            return

        # Ensures the point missed is no longer a candidate.
        del self._point_map[ship][point]

        # TODO: I don't believe the set is really needed here; this could likely be a comprehension.
        candidates = set(
            point
            for point in self._points_available_all(
                point,
                self.game.dimensions,
                ship_length,
                set(self._point_map[ship]) | self._ship_hits[ship],
            )
            if point in self._point_map[ship]
        )

        for candidate in candidates:
            permutations = self._possible_placements_2(
                self.game,
                candidate,
                ship,
                set(self._point_map[ship]) | self._ship_hits[ship],
            )
            if permutations:
                self._point_map[ship][candidate] = permutations
            else:
                del self._point_map[ship][candidate]

    def _progate_certainties(self) -> bool:
        # Check for the last point left for each ship, and propagate this info to other ships.
        # i = 0
        while True:
            # i += 1
            success = False
            for ship, point_map in self._point_map.items():
                points_remaining = self.game.ships[ship] - len(self._ship_hits[ship])
                if not points_remaining:
                    continue

                total_permutations = sum(point_map.values())
                point_probabilities = (
                    (point, (points_remaining * (permutations / total_permutations)))
                    for point, permutations in point_map.items()
                )

                for candidate, probability in point_probabilities:
                    if probability != 1.0:
                        continue

                    if candidate in self._points_deduced:
                        continue

                    self._points_deduced.add(candidate)

                    for ship_inner, ship_length in (
                        (s, l) for s, l in self.game.ships.items() if s != ship
                    ):
                        if candidate in self._point_map[ship_inner]:
                            # print(
                            #     f"--> {ship} / {ship_inner} - {candidate}, {self._point_map[ship_inner].get(candidate, None)}"
                            # )
                            self._evaluate_miss(ship_inner, ship_length, candidate)
                            success = True
            if not success:
                break

            # if 1 < i:
            #     print("Double plus good")
        return success

    def _update_point_ranking(self):
        self._ranked_points.clear()

        points = {}
        for ship, point_permutations in self._point_map.items():
            if not point_permutations:
                continue

            total_permutations = 0
            for point, permutations in point_permutations.items():
                if point not in points:
                    points[point] = [0, 0.0]
                points[point][0] += permutations
                total_permutations += permutations

            remaining_points = self.game.ships[ship] - len(self._ship_hits[ship])
            for point, permutations in point_permutations.items():
                point_prob = remaining_points * (permutations / total_permutations)
                points[point][1] = 1 - ((1 - points[point][1]) * (1 - point_prob))

        self._ranked_points.extend(
            (point, permutations, probability)
            for point, (permutations, probability) in points.items()
        )

        # Sort by permutations
        self._ranked_points.sort(key=lambda v: v[1], reverse=True)

        # Sort by probability
        self._ranked_points.sort(key=lambda v: v[2], reverse=True)

    def points_remain(self) -> bool:
        return 0 < len(self._ranked_points)

    def get_point(self) -> Point:
        _top_points = self.top_points_with_score()
        return random.choice(tuple(v[0] for v in _top_points))

    def top_points(self) -> Iterator[Point]:
        _top_points = self.top_points_with_score()
        top_points = (v[0] for v in _top_points)
        return top_points

    def top_points_with_score(self) -> Iterator[Tuple[Point, int, float]]:
        # Group the points according to their score.
        grouped_points = (
            group
            for permutations, group in groupby(
                self._ranked_points, key=lambda v: (v[1], v[2])
            )
        )
        top_points = next(grouped_points)
        return top_points

    @classmethod
    def _initial_candidates(cls, game: GameConfig) -> Dict[str, Dict[Point, int]]:
        candidates = {}

        for ship_id, ship_length in game.ships.items():
            candidates[ship_id] = {}
            for point in game.points:
                candidates[ship_id][point] = cls._possible_placements(
                    game, point, ship_id, set()
                )

        return candidates

    @classmethod
    def _initial_estimates(
        cls, game: GameConfig
    ) -> Dict[str, Dict[Point, PointEstimate]]:
        candidates = {}

        for ship_id in game.ships:
            candidates[ship_id] = {}
            for point in game.points:
                permutations = cls._possible_placements(game, point, ship_id, set())
                candidates[ship_id][point] = PointEstimate(permutations, 0.0)

        for ship_id, ship_length in game.ships.items():
            remaining_points = ship_length
            total_permutations = sum(
                e.permutations for e in candidates[ship_id].values()
            )
            scale_factor = remaining_points / total_permutations
            for estimate in candidates[ship_id].values():
                estimate.probability = scale_factor * estimate.permutations

        return candidates

    @classmethod
    def _placement_probabilities(
        cls, remaining_points: int, placement_map: Dict[Point, int]
    ) -> Iterator[Tuple[Point, float]]:
        total_placements = sum(placement_map.values())
        scale_factor = remaining_points / total_placements
        return ((p, (scale_factor * v)) for p, v in placement_map.items())

    @classmethod
    def _initial_odds(
        cls, game: GameConfig, candidates: Dict[str, Dict[Point, float]]
    ) -> Board:
        odds_board = Board(game.rows, game.cols, initial_point=0)

        for point_score_map in candidates.values():
            for (row, col), score in point_score_map.items():
                odds_board[row][col] += score

        return odds_board

    @classmethod
    def _points_in_direction(
        cls,
        point: Point,
        direction: Orientation,
        dimensions: Tuple[int, int],
        ship_length: int,
    ) -> Iterator[Point]:
        row, col = point
        rows, cols = dimensions

        if direction is Orientation.Up:
            _range = range(row - 1, row - ship_length, -1)
            points = ((r, col) for r in _range if 0 <= r < rows)
        elif direction is Orientation.Down:
            _range = range(row + 1, row + ship_length)
            points = ((r, col) for r in _range if 0 <= r < rows)
        elif direction is Orientation.Left:
            _range = range(col - 1, col - ship_length, -1)
            points = ((row, c) for c in _range if 0 <= c < cols)
        else:
            _range = range(col + 1, col + ship_length)
            points = ((row, c) for c in _range if 0 <= c < cols)

        for neighbor_point in points:
            yield neighbor_point

    @classmethod
    def _points_available(
        cls,
        point: Point,
        direction: Orientation,
        dimensions: Tuple[int, int],
        ship_length: int,
        points_tested: Set[Point],
    ) -> Iterator[Point]:
        """
        Returns points non-inclusive of the given point.
        """
        points = cls._points_in_direction(point, direction, dimensions, ship_length)
        for neighbor_point in points:
            if neighbor_point in points_tested:
                break
            yield neighbor_point

    @classmethod
    def _points_available_2(
        cls,
        point: Point,
        direction: Orientation,
        dimensions: Tuple[int, int],
        ship_length: int,
        available: Set[Point],
    ) -> Iterator[Point]:
        """
        Returns points non-inclusive of the given point.
        The points returned are found in the `available` set.
        """
        points = cls._points_in_direction(point, direction, dimensions, ship_length)
        for neighbor_point in points:
            if neighbor_point not in available:
                break
            yield neighbor_point

    @classmethod
    def _points_available_all(
        cls,
        point: Point,
        dimensions: Tuple[int, int],
        ship_length: int,
        available: Set[Point],
    ) -> Iterator[Point]:
        """
        Returns points non-inclusive of the given point, for all Orientations.
        The points returned are found in the `available` set.
        """
        for direction in Orientation:
            yield from cls._points_available_2(
                point, direction, dimensions, ship_length, available
            )

    @classmethod
    def _possible_placements_2(
        cls,
        game: GameConfig,
        point: Point,
        ship_id: str,
        available: Set[Point],
    ) -> int:
        """The number of ways a ship of given length may be placed over a point.

        The number of possible placements is equal to the sum of the number of possible
        vertical and horizontal placements.
        """
        assert 0 <= point[0] < game.rows, "Row value out-of-bounds"
        assert 0 <= point[1] < game.cols, "Col value out-of-bounds"

        row, col = point
        ship_length = game.ships[ship_id]

        # Vertical
        ## Up
        available_up = sum(
            1
            for _ in cls._points_available_2(
                point,
                Orientation.Up,
                game.dimensions,
                ship_length,
                available,
            )
        )

        ## Down
        available_down = sum(
            1
            for _ in cls._points_available_2(
                point,
                Orientation.Down,
                game.dimensions,
                ship_length,
                available,
            )
        )

        ## Vertical total
        available_vertical = available_up + available_down
        placements_vertical = max(available_vertical - ship_length + 2, 0)

        # Horizontal
        ## Left
        available_left = sum(
            1
            for _ in cls._points_available_2(
                point,
                Orientation.Left,
                game.dimensions,
                ship_length,
                available,
            )
        )

        ## Right
        available_right = sum(
            1
            for _ in cls._points_available_2(
                point,
                Orientation.Right,
                game.dimensions,
                ship_length,
                available,
            )
        )

        ## Horizontal total
        available_horizontal = available_left + available_right
        placements_horizontal = max(available_horizontal - ship_length + 2, 0)

        return placements_vertical + placements_horizontal

    @classmethod
    def _possible_placements(
        cls,
        game: GameConfig,
        point: Point,
        ship_id: str,
        points_tested: Set[Point],
    ) -> int:
        """The number of ways a ship of given length may be placed over a point.

        The number of possible placements is equal to the sum of the number of possible
        vertical and horizontal placements.
        """
        assert 0 <= point[0] < game.rows, "Row value out-of-bounds"
        assert 0 <= point[1] < game.cols, "Col value out-of-bounds"

        row, col = point
        ship_length = game.ships[ship_id]

        # Vertical
        ## Up
        # available_up = cls._available_points_up(point, game, ship_length, points_tested)
        available_up = sum(
            1
            for _ in cls._points_available(
                point,
                Orientation.Up,
                game.dimensions,
                ship_length,
                points_tested,
            )
        )

        ## Down
        # available_down = cls._available_points_down(point, game, ship_length, points_tested)
        available_down = sum(
            1
            for _ in cls._points_available(
                point,
                Orientation.Down,
                game.dimensions,
                ship_length,
                points_tested,
            )
        )

        ## Vertical total
        available_vertical = available_up + available_down
        placements_vertical = max(available_vertical - ship_length + 2, 0)

        # Horizontal
        ## Left
        # available_left = cls._available_points_left(point, game, ship_length, points_tested)
        available_left = sum(
            1
            for _ in cls._points_available(
                point,
                Orientation.Left,
                game.dimensions,
                ship_length,
                points_tested,
            )
        )

        ## Right
        # available_right = cls._available_points_right(point, game, ship_length, points_tested)
        available_right = sum(
            1
            for _ in cls._points_available(
                point,
                Orientation.Right,
                game.dimensions,
                ship_length,
                points_tested,
            )
        )

        ## Horizontal total
        available_horizontal = available_left + available_right
        placements_horizontal = max(available_horizontal - ship_length + 2, 0)

        return placements_vertical + placements_horizontal


class ProbabilityStrategy(BattleShipStrategy):
    def __init__(self, game_config: GameConfig, enemy: Enemy):
        super().__init__(game_config, enemy)
        self._points_tested = set()
        self._ps = ProbabilitySearch(game_config)
        self._turns: int = 0

    def step(
        self, verbose=False
    ) -> Union[Tuple[Point, bool, bool, Union[str, None]], None]:
        if self._ps.points_remain() and not self.enemy.all_sunk():
            point = self._ps.get_point()
            is_hit, is_sunk, ship_id = self.enemy.apply_missile(point)

            if verbose:
                print(
                    f"Point: {point}, {'HIT, ' if is_hit else 'MISS'}{ship_id if is_hit else ''}{', SUNK' if is_sunk else ''}"
                )
                print(
                    f"Ships Remaining: {', '.join(s.id for s in self.enemy._ships if not s.is_sunk())}"
                )
                print(f"Ships Hit: {self.enemy.ships_hit()}")

            if is_hit:
                self._ps.hit(point, ship_id, verbose=verbose)
            else:
                self._ps.miss(point, verbose=verbose)

            if verbose:
                print(f"{self._ps.probability_board}")
                print(f"{'-'*40}")

            self._turns += 1

            return (point, is_hit, is_sunk, ship_id)
        return None

    def solve(self, verbose=False):
        # While not solved or until no points are left to test:
        # - Of the set of points most likely to intersect a ship,
        #   randomly pick a point to test.
        # - Test the point to see if it hits a ship.
        #   - If hit, proceed with the kill strategy.
        #   - If not hit, continue loop.

        if verbose:
            print(f"-----\n{repr(self._ps.probability_board)}\n")

        while self._ps.points_remain() and not self.enemy.all_sunk():
            # top_points = tuple(self._ps.top_points())
            # point = random.choice(top_points)
            point = self._ps.get_point()
            is_hit, is_sunk, ship_id = self.enemy.apply_missile(point)

            if verbose:
                print(
                    f"Point: {point}, {'HIT, ' if is_hit else 'MISS'}{ship_id if is_hit else ''}{', SUNK' if is_sunk else ''}"
                )
                print(
                    f"Ships Remaining: {', '.join(s.id for s in self.enemy._ships if not s.is_sunk())}"
                )
                print(f"Ships Hit: {self.enemy.ships_hit()}")

            if is_hit:
                self._ps.hit(point, ship_id, verbose=verbose)
            else:
                self._ps.miss(point, verbose=verbose)

            if verbose:
                print(f"{self._ps.probability_board}")
                print(f"{'-'*40}")

            self._turns += 1

        if verbose:
            print(f"Finished: {self._turns} turns")

    def is_solved(self) -> bool:
        return all(s.is_sunk() for s in self.enemy._ships)

    def solution(self):
        return super().solution()

    def turns(self) -> int:
        return self._turns


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
