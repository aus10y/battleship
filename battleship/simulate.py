import itertools
from timeit import default_timer as timer
from typing import Callable, Dict, List, Set, Tuple, Type

from .board import (
    Point,
    Ship,
    format_board,
    generate_ship_placements,
    initialize_board,
    set_board,
)
from .enemy import Enemy
from .game import GameConfig
from .strategy import (
    BattleShipStrategy,
    hit_strategy_full_context,
    random_checkerboard_search,
)
from .tools import enumerate_all_points


DEBUG = False


def simulate_game(
    dimensions: Tuple[int, int],
    ship_sizes: Tuple[int, ...],
    StrategyClass: Type[BattleShipStrategy],
):
    game = GameConfig(dimensions, ship_sizes)
    ships = generate_ship_placements(game)
    enemy = Enemy(game.dimensions, ships)
    strategy = StrategyClass(game, enemy)
    return strategy.solve()

def simulate_games(
    dimensions: Tuple[int, int],
    ship_sizes: Tuple[int, ...],
    StrategyClass: Type[BattleShipStrategy],
    games: int = 1000
):
    turns = 0
    solved = 0
    game = GameConfig(dimensions, ship_sizes)
    for _ in range(games):
        ships = generate_ship_placements(game)
        enemy = Enemy(game.dimensions, ships)
        strategy = StrategyClass(game, enemy)
        strategy.solve()
        if strategy.is_solved():
            solved += 1
        turns += len(strategy._ps._points_tested)
    return solved, turns / games


def play_one_game(rows, cols, ships):
    game = GameConfig((rows, cols), ships)
    ship_placements = generate_ship_placements(game)
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


def _ships_remain(ships: Tuple[Ship, ...]) -> bool:
    # return any(p.points for p in ship_placements)
    return any(not ship.is_sunk() for ship in ships)


def run_simulation(
    dimensions: Tuple[int, int],
    ships: Tuple[Ship, ...],
    search_strategy: Callable,
    hit_strategy: Callable,
) -> Tuple[bool, List[Point], Set[Point]]:
    rows, cols = dimensions

    turns = []
    points_remaining = set(enumerate_all_points(rows, cols))

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
    game = GameConfig((rows, cols), ships)
    bins = dict()
    start_time = timer()
    for _ in range(runs):
        ship_placements = generate_ship_placements(game)
        board = initialize_board(game.dimensions)
        set_board(board, ship_placements)

        sank_all, turns, remaining = run_simulation(
            game.dimensions,
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
