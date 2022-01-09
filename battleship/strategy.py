from enum import Enum
from itertools import groupby
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


# -----------------------------------------------------------------------------


class PointStatus(Enum):
    Unknown = None
    Miss = 1
    Hit = 2


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
        self._point_placement_map: Dict[
            str, Dict[Point, int]
        ] = self._initial_candidates(game)
        self._point_probability_map: Dict[str, Dict[Point, float]] = {
            ship: dict(
                self._placement_probabilities(
                    ship_length, self._point_placement_map[ship]
                )
            )
            for ship, ship_length in self.game.ships.items()
        }
        self._probability_board = self._initial_odds(
            self.game, self._point_probability_map
        )
        self._hits: Set[Point] = set()
        self._ship_hits: Dict[str, Set[Point]] = {
            ship_id: set() for ship_id in self.game.ships
        }
        self._misses: Set[Point] = set()
        self._ranked_points = []
        self._update_point_ranking()

    @property
    def _points_tested(self):
        return self._hits | self._misses

    def hit(self, point_hit: Point, ship_hit: str, verbose=False):
        row, col = point_hit
        self._hits.add(point_hit)
        self._ship_hits[ship_hit].add(point_hit)

        # Remove the hit point from the candidacy of all ships.
        for _ship in self.game.ships:
            if point_hit in self._point_placement_map[_ship]:
                self._probability_board[row][col] -= self._point_probability_map[_ship][
                    point_hit
                ]
                del self._point_placement_map[_ship][point_hit]
                del self._point_probability_map[_ship][point_hit]

        # Subtract off the contribution made from each candidate point for this ship.
        # The points are not yet removed from the set of candidates for this ship.
        for point, score in self._point_probability_map[ship_hit].items():
            _row, _col = point
            self._probability_board[_row][_col] -= score

        # -------------------
        # General Logic
        #
        # For each ship, find the points local to the point hit.
        #
        # For the ship hit, intersect the local points with the existing
        # candidates and clear the existing candidates.
        #
        # For each ship, calculate the placements for each local point and
        # store the points back to the respective candidate dictionaries if the
        # value is not zero.
        # -------------------

        # For each ship, find the points local to the point hit.

        def _calc_local_points(ship: str) -> Set[Point]:
            return set(
                point
                for point in self._points_available_all(
                    point_hit,
                    self.game.dimensions,
                    self.game.ships[ship],
                    set(self._point_placement_map[ship].keys()) | self._ship_hits[ship],
                )
                if point not in self._ship_hits[ship]
            )

        local_points_by_ship = {
            s: _calc_local_points(s) for s in self.game.ships if s != ship_hit
        }

        # For the ship hit, intersect the local points with the existing
        # candidates and clear the existing candidates.

        _local_points = _calc_local_points(ship_hit) & set(
            self._point_placement_map[ship_hit].keys()
        )
        _candidate_points: Dict[Point, int] = {}
        self._point_placement_map[ship_hit].clear()
        self._point_probability_map[ship_hit].clear()
        for point in _local_points:
            _row, _col = point

            # When calculating the placements, the candidates should include
            # points already hit for the current (loop) ship.
            placements = self._possible_placements_2(
                self.game,
                point,
                ship_hit,
                set(_local_points) | self._ship_hits[ship_hit],
            )

            if placements:
                _candidate_points[point] = placements

        # Calculate odds for remaining points
        num_points_remaining = self.game.ships[ship_hit] - len(
            self._ship_hits[ship_hit]
        )
        total_placements = sum(_candidate_points.values())
        if total_placements:
            scale_factor = (num_points_remaining) / total_placements
            for point, placements in _candidate_points.items():
                _row, _col = point
                odds = scale_factor * placements
                self._point_placement_map[ship_hit][point] = placements
                self._point_probability_map[ship_hit][point] = odds
                self._probability_board[_row][_col] += odds

        # For each remaining ship, calculate the placements for each local point and
        # store the points back to the respective candidate dictionaries if the
        # value is not zero.

        for ship, points in (
            (s, p) for s, p in local_points_by_ship.items() if s != ship_hit
        ):
            # Accumulate points that have possible placements

            _candidate_points.clear()
            for point in points:
                _row, _col = point
                self._probability_board[_row][_col] -= self._point_probability_map[
                    ship
                ][point]

                # When calculating the placements, the candidates should include
                # points already hit for the current ship (of the loop).
                placements = self._possible_placements_2(
                    self.game,
                    point,
                    ship,
                    set(self._point_placement_map[ship]) | self._ship_hits[ship],
                )

                if placements:
                    _candidate_points[point] = placements
                    self._point_placement_map[ship][point] = placements
                else:
                    if point in self._point_placement_map[ship]:
                        del self._point_placement_map[ship][point]
                        del self._point_probability_map[ship][point]

        def _scale_factor(s):
            return (self.game.ships[s] - len(self._ship_hits[s])) / sum(
                self._point_placement_map[s].values()
            )

        # scale_factor = (self.game.ships[ship] - len(self._ship_hits[ship])) / sum(
        #     self._point_placement_map[ship].values()
        # )

        ship_scales = {
            ship: _scale_factor(ship) for ship in self.game.ships
            if self._point_placement_map[ship]
        }

        for ship in ship_scales:
            for point, placements in self._point_placement_map[ship].items():
                #prev_odds = self._point_probability_map[ship][point]
                _row, _col = point
                odds = ship_scales[ship] * placements
                # self._point_placement_map[ship][point] = placements
                self._point_probability_map[ship][point] = odds
                #self._probability_board[_row][_col] += odds - prev_odds

        for point in self._probability_board.points():
            self._probability_board.set(
                point,
                sum(
                    prob[point]
                    for prob in self._point_probability_map.values()
                    if point in prob
                ),
            )

        self._update_point_ranking()

    def miss(self, point: Point, verbose=False):
        row, col = point
        self._misses.add(point)
        self._probability_board[row][col] = 0

        # Need to update odds with the candidates
        potential_candidates = set()
        for ship, ship_length in self.game.ships.items():
            if point not in self._point_placement_map[ship]:
                continue

            del self._point_placement_map[ship][point]
            del self._point_probability_map[ship][point]

            for orientation in Orientation:
                points = list(
                    # self._points_available(
                    self._points_in_direction(
                        point,
                        orientation,
                        self.game.dimensions,
                        ship_length,
                        # points_tested,
                    )
                )
                potential_candidates.update(points)

        candidates = set()
        for ship in self.game.ships:
            if not self._point_placement_map[ship]:
                continue

            # Subtract off the current probabilities
            for point, probability in self._point_probability_map[ship].items():
                _row, _col = point
                self._probability_board[_row][_col] -= probability

            # Find the candidate points
            ship_candidates = potential_candidates & set(
                self._point_placement_map[ship].keys()
            )
            candidates.update(ship_candidates)
            for candidate in ship_candidates:
                candidate_row, candidate_col = candidate
                placements = self._possible_placements_2(
                    self.game,
                    candidate,
                    ship,
                    set(self._point_placement_map[ship]) | self._ship_hits[ship],
                )
                if placements:
                    self._point_placement_map[ship][candidate] = placements
                else:
                    del self._point_placement_map[ship][candidate]
                    del self._point_probability_map[ship][candidate]

            if not self._point_placement_map[ship]:
                continue

            # Determine the probabilities
            self._point_probability_map[ship].clear()
            self._point_probability_map[ship].update(
                self._placement_probabilities(
                    (self.game.ships[ship] - len(self._ship_hits[ship])),
                    self._point_placement_map[ship],
                )
            )

            # Set the probabilities
            for point, probability in self._point_probability_map[ship].items():
                _row, _col = point
                self._probability_board[_row][_col] += probability

        # Update the odds board
        # for candidate in candidates:
        #     candidate_row, candidate_col = candidate
        #     self._probability_board[candidate_row][candidate_col] = 0

        #     for ship, candidate_map in self._point_placement_map.items():
        #         if candidate not in candidate_map:
        #             continue
        #         self._probability_board[candidate_row][candidate_col] += candidate_map[
        #             candidate
        #         ]

        self._update_point_ranking()

    def sunk(self, ship_id: str):
        # TODO: Remove all remaining candidates when a ship is sunk.
        pass

    def _update_point_ranking(self):
        rankings = [
            (point, score)
            for (point, score) in self._probability_board.items()
            if score
        ]

        # Sort according to odds, descending
        rankings.sort(key=lambda v: v[1], reverse=True)
        self._ranked_points = rankings

    def points_remain(self) -> bool:
        return 0 < len(self._ranked_points)

    def get_point(self) -> Point:
        _top_points = self.top_points_with_score()
        return random.choice(tuple(v[0] for v in _top_points))

    def top_points(self) -> Iterator[Point]:
        _top_points = self.top_points_with_score()
        top_points = (v[0] for v in _top_points)
        return top_points

    def top_points_with_score(self) -> Iterator[Tuple[Point, int]]:
        # Group the points according to their score.
        grouped_points = (
            group for score, group in groupby(self._ranked_points, key=lambda v: v[1])
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
            # total_placements = sum(candidates[ship_id].values())
            # scale = ship_length / total_placements
            # candidates[ship_id] = {
            #     point: (scale * placements)
            #     for point, placements in candidates[ship_id].items()
            # }

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


"""
class _ProbabilitySearch:
    # Thoughts for ranking, updating, and selecting points:
    # - Have a function that takes a board, having one or more points placed,
    # and scores the points and returns the points ranked it some manner.
    # - To select a point for firing at the enemy, choose the point having the
    # highest rank/score. If more than one point has the same highest rank, choose
    # from the points at random.
    # - To efficiently update the rankings after testing a point:
    #   - If the point does not hit a ship:
    #     - Re-rank all cardinal neighbors that are within reach of the largest
    #     ship and within the boundaries.
    #   - If the point does hit a ship:
    #     - Proceed with the kill strategy. When choosing a neighboring point to
    #     test in order to sink the ship, the points should be chosen according to
    #     their scores. Where neighboring points have the same rank, choose
    #     between them at random.
    #   - When a ship is sunk:
    #     - All remaining points need to be re-ranked due to their being one fewer
    #     ship.
    # - Data structures for ranking and tracking points:
    #   - A List of (point, score) tuples. The list would be sorted by score
    #   descending.
    #   - A dictionary, mapping points to the points index in the ranking list.

    def __init__(self, game: GameConfig):
        self.game = game
        self._ships_remaining = self.game.ships
        # self._odds_board = Board(game.rows, game.cols, initial_point=0)
        self._odds_board = self._calculate_initial_odds(self.game)
        # self._hits_board = Board(
        #     game.rows, game.cols, initial_point=PointStatus.Unknown
        # )
        self._ranked_points: List[Tuple[Point, int]] = []
        self._points_tested: Set[Point] = set()
        self._hits: Dict[str, Set[Point]] = {
            ship_id: set() for ship_id in self.game.ships.keys()
        }
        self._hit_candidates: Dict[str, Set[Point]] = {
            ship_id: set() for ship_id in self.game.ships.keys()
        }
        self._hits_all: Set[Point] = set()
        self._misses: Set[Point] = set()

        self._calculate_ship_odds()
        self._update_point_ranking()

    def points_remain(self) -> bool:
        return 0 < len(self._ranked_points)

    def top_points(self) -> Iterator[Point]:
        _top_points = self.top_points_with_score()
        top_points = (v[0] for v in _top_points)
        return top_points

    def top_points_with_score(self) -> Iterator[Tuple[Point, int]]:
        # Group the points according to their score.
        grouped_points = (
            group for score, group in groupby(self._ranked_points, key=lambda v: v[1])
        )
        top_points = next(grouped_points)
        return top_points

    def _algo(self):
        # For each ship:
        # - Maintain a set of "candidate" points.
        #   - Each point could have the current possibilities paired with it.
        # - At the beginning of the game, each ship will have all points as candidates.
        # For each miss:
        # - Loop over the ships:
        #   - For each neighboring point (to the point missed), subtract off the
        #   previous candidate "score".
        #   - Recalculate the new score for the points, and update the candidates set.
        # For each hit:
        # - For the ship hit:
        #   - Subtract off the contribution from each candidate.
        #   - Remove all Candidates
        #   - Determinie all neighboring points
        #   - Score the points and store them as candidates.
        pass

    def hit(self, point_hit: Point, ship: str):
        # Check if this is the first hit for the given ship:
        if not self._hits[ship]:
            # Since this is the first hit for the given ship:
            # - For all points other than the one hit:
            #   - Subtract off the possible placements for the ship hit, not taking
            #   into account the point just hit (we are subtracting the previous
            #   beliefs for the given ship).
            for point in self._odds_board.points():
                row, col = point

                # Account for hte fact that the point hit now has no other
                # possibilities.
                if point == point_hit:
                    self._odds_board[row][col] = 0
                    continue

                # No need to perform any calculations for points already tested.
                # Whether the tested points were hits or misses, we don't need
                # to perform any calculation.
                if point in self._points_tested:
                    continue

                # Finally, subtract off the given ships possibilities.
                self._odds_board[row][col] -= self._possible_placements(
                    self.game, point, ship, self._points_tested
                )
        else:
            # Since this is not the first hit or the given ship:
            # - For all existing "candidate" points (neighboring points that
            # could potentially be future hits for the given ship), subtract off
            # the possible placements (subtract off our previous beliefs).
            pass

            # - Find the candidate points for the point hit.
            # - Take the intersection of the new candidates and the existing
            # candidates.
            # - Calculate the possibilities for the intersecting candidates.
            pass

        # For all other ships that have no hits
        hidden_ships = (
            s for s, hits in self._hits.items() if not hits and s is not ship
        )
        for ship_id in hidden_ships:
            # For points that may be reached from this hit:

            # Subtract off the old possibilities for this ship.
            pass

            # Add the new possibilities.
            pass

        # row, col = point
        # self._hits[ship_id].add(point)
        # self._hits_all.add(point)
        # self._points_tested.add(point)
        # self._odds_board[row][col] = ship_id
        # self._hits_board[row][col] = PointStatus.Hit
        # self._calculate_ship_odds()
        # self._update_point_ranking()

    def miss(self, point: Point):
        # Subtract odds from local points.
        row, col = point
        self._misses.add(point)
        self._points_tested.add(point)
        self._odds_board[row][col] = 0
        self._hits_board[row][col] = PointStatus.Miss
        self._calculate_ship_odds()
        self._update_point_ranking()

    def sunk(self, ship_id: str):
        pass

    @classmethod
    def _calculate_initial_odds(cls, game: GameConfig):
        odds_board = Board(game.rows, game.cols, initial_point=0)

        for (row, col) in odds_board.points():
            for ship_id, _ in game.ships.items():
                odds_board[row][col] += cls._possible_placements(
                    game, (row, col), ship_id, set()
                )

        return odds_board

    # def _calculate_ship_odds(self):
    #     hits = self._hits
    #     misses = self._misses

    #     for row, col in self._odds_board.points():
    #         self._odds_board[row][col] = 0
    #         for ship_length in self._ships_remaining.values():
    #             point = (row, col)
    #             self._odds_board[row][col] += self._possible_placements(
    #                 self._hits_board, point, ship_length
    #             )

    def _calculate_ship_odds(self):
        for point in self._odds_board.points():
            row, col = point

            if point in self._points_tested:
                continue

            self._odds_board[row][col] = 0
            for ship_id, ship_length in self.game.ships.items():
                # If point is remote from a hit for the given ship, skip
                ship_hits = self._hits[ship_id]
                if ship_hits and self._is_remote(point, ship_length, ship_hits):
                    continue

                self._odds_board[row][col] += self._possible_placements(
                    self.game,
                    (row, col),
                    ship_id,
                    self._misses.union(self._hits[ship_id]),
                )

    def _update_point_ranking(self):
        rankings = []
        for point in self._odds_board.points():
            if point in self._points_tested:
                continue
            row, col = point
            odds = self._odds_board[row][col]
            if odds:
                rankings.append((point, self._odds_board[row][col]))

        # Sort according to odds, descending
        rankings.sort(key=lambda v: v[1], reverse=True)
        self._ranked_points = rankings

    @classmethod
    def _points_until_blocked(
        cls, points: Iterable[Point], points_tested: Set[Point]
    ) -> int:
        available = 0
        for point in points:
            if point in points_tested:
                break
            available += 1
        return available

    @staticmethod
    def _is_remote(point: Point, ship_length: int, ship_hits: Set[Point]) -> bool:
        # If there are no hits, then we cannot consider the point to be remote.
        if len(ship_hits) == 0:
            return False

        # Remote (True) if any point lies outside the row & col of the given point.
        if any((point[0] != p[0] and point[1] != p[1] for p in ship_hits)):
            return True

        # Remote (True) if any hit lies more than (ship_length - 1) away from the given point.
        for hit in ship_hits:
            if point[0] == hit[0]:
                # Row is same
                if (ship_length - 1) < abs(point[1] - hit[1]):
                    return True
            elif point[1] == hit[1]:
                # Col is same
                if (ship_length - 1) < abs(point[0] - hit[0]):
                    return True
            else:
                return True

        return False

    @classmethod
    def _available_points_up(
        cls, point: Point, game: GameConfig, ship_length: int, points_tested: Set[Point]
    ):
        row, col = point
        points = [
            (r, col)
            for r in range(row - 1, row - ship_length, -1)
            if 0 <= r < game.rows
        ]
        return cls._points_until_blocked(points, points_tested)

    @classmethod
    def _available_points_down(
        cls, point: Point, game: GameConfig, ship_length: int, points_tested: Set[Point]
    ):
        row, col = point
        points = [
            (r, col) for r in range(row + 1, row + ship_length) if 0 <= r < game.rows
        ]
        return cls._points_until_blocked(points, points_tested)

    @classmethod
    def _available_points_left(
        cls, point: Point, game: GameConfig, ship_length: int, points_tested: Set[Point]
    ):
        row, col = point
        points = [
            (row, c)
            for c in range(col - 1, col - ship_length, -1)
            if 0 <= c < game.cols
        ]
        return cls._points_until_blocked(points, points_tested)

    @classmethod
    def _available_points_right(
        cls, point: Point, game: GameConfig, ship_length: int, points_tested: Set[Point]
    ):
        row, col = point
        points = [
            (row, c) for c in range(col + 1, col + ship_length) if 0 <= c < game.cols
        ]
        return cls._points_until_blocked(points, points_tested)

    @classmethod
    def _possible_placements(
        cls,
        game: GameConfig,
        point: Point,
        ship_id: str,
        points_tested: Set[Point],
    ) -> int:
        # The number of ways a ship of given length may be placed over a point.

        # The number of possible placements is equal to the sum of the number of possible
        # vertical and horizontal placements.
        assert 0 <= point[0] < game.rows, "Row value out-of-bounds"
        assert 0 <= point[1] < game.cols, "Col value out-of-bounds"

        row, col = point
        ship_length = game.ships[ship_id]

        # Vertical
        ## Up
        available_up = cls._available_points_up(point, game, ship_length, points_tested)

        ## Down
        available_down = cls._available_points_down(
            point, game, ship_length, points_tested
        )

        ## Vertical total
        available_vertical = available_up + available_down
        placements_vertical = max(available_vertical - ship_length + 2, 0)

        # Horizontal
        ## Left
        available_left = cls._available_points_left(
            point, game, ship_length, points_tested
        )

        ## Right
        available_right = cls._available_points_right(
            point, game, ship_length, points_tested
        )

        ## Horizontal total
        available_horizontal = available_left + available_right
        placements_horizontal = max(available_horizontal - ship_length + 2, 0)

        return placements_vertical + placements_horizontal

    # @classmethod
    # def _possible_placements_1(
    #     cls,
    #     board: Board,
    #     point: Point,
    #     ship_length: int,
    # ) -> int:
    #     # The number of ways a ship of given length may be placed over a point.
    #     #
    #     # The number of possible placements is equal to the sum of the number of possible
    #     # vertical and horizontal placements.

    #     assert 0 <= point[0] < board.rows, "Row value out-of-bounds"
    #     assert 0 <= point[1] < board.cols, "Col value out-of-bounds"

    #     # TODO: Should skip possible placements (return 0) if the given point is
    #     # local to a hit for a ship of the given length.
    #     pass

    #     row, col = point

    #     # Vertical
    #     ## Up
    #     points = [
    #         (r, col)
    #         for r in range(row - 1, row - ship_length, -1)
    #         if 0 <= r < board.rows
    #     ]
    #     available_up = cls._points_until_blocked(board, points)

    #     ## Down
    #     points = [
    #         (r, col) for r in range(row + 1, row + ship_length) if 0 <= r < board.rows
    #     ]
    #     available_down = cls._points_until_blocked(board, points)

    #     ## Vertical total
    #     available_vertical = available_up + available_down
    #     placements_vertical = max(available_vertical - ship_length + 2, 0)

    #     # Horizontal
    #     ## Left
    #     points = [
    #         (row, c)
    #         for c in range(col - 1, col - ship_length, -1)
    #         if 0 <= c < board.cols
    #     ]
    #     available_left = cls._points_until_blocked(board, points)

    #     ## Right
    #     points = [
    #         (row, c) for c in range(col + 1, col + ship_length) if 0 <= c < board.cols
    #     ]
    #     available_right = cls._points_until_blocked(board, points)

    #     ## Horizontal total
    #     available_horizontal = available_left + available_right
    #     placements_horizontal = max(available_horizontal - ship_length + 2, 0)

    #     return placements_vertical + placements_horizontal
    """


class ProbabilityStrategy(BattleShipStrategy):
    def __init__(self, game_config: GameConfig, enemy: Enemy):
        super().__init__(game_config, enemy)
        self._points_tested = set()
        self._ps = ProbabilitySearch(game_config)

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
                print(f"{self._ps._probability_board}")
                print(f"{'-'*40}")

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
            print(f"-----\n{repr(self._ps._probability_board)}\n")

        i = 0
        while self._ps.points_remain() and not self.enemy.all_sunk():
            i += 1
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
                print(f"{self._ps._probability_board}")
                print(f"{'-'*40}")

        if verbose:
            print(f"Finished: {i} steps")

    def is_solved(self) -> bool:
        return all(s.is_sunk() for s in self.enemy._ships)

    def solution(self):
        return super().solution()


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
