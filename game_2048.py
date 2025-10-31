"""
Stateless 2048 Game Implementation
Pure functional approach with no classes.
"""

import random
from typing import List
from copy import deepcopy


def init_grid() -> List[List[int]]:
    """
    Initialize a new 4x4 grid with two random tiles (2 or 4).
    
    Returns:
        4x4 grid (list of lists) with two random tiles placed
    """
    grid = [[0] * 4 for _ in range(4)]
    
    # Add first random tile (90% chance of 2, 10% chance of 4)
    i, j = random.randint(0, 3), random.randint(0, 3)
    grid[i][j] = 2 if random.random() < 0.9 else 4
    
    # Add second random tile at a different position
    while True:
        i, j = random.randint(0, 3), random.randint(0, 3)
        if grid[i][j] == 0:
            grid[i][j] = 2 if random.random() < 0.9 else 4
            break
    
    return grid


def _add_random_tile(grid: List[List[int]]) -> None:
    """
    Add a random tile (2 with 90% probability or 4 with 10% probability) 
    to an empty position in the grid.
    Modifies the grid in place.
    
    Args:
        grid: 4x4 grid to add tile to
    """
    empty_positions = [
        (i, j) for i in range(4) for j in range(4) if grid[i][j] == 0
    ]
    if empty_positions:
        i, j = random.choice(empty_positions)
        grid[i][j] = 2 if random.random() < 0.9 else 4


def _merge_line(line: List[int]) -> List[int]:
    """
    Merge a single line (row or column) to the left.
    
    Args:
        line: List of 4 integers representing a row or column
    
    Returns:
        Merged line with values shifted left
    """
    # Remove zeros
    non_zero = [val for val in line if val != 0]
    
    # Merge adjacent equal values
    merged = []
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged.append(non_zero[i] * 2)
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1
    
    # Fill with zeros to maintain length of 4
    merged.extend([0] * (4 - len(merged)))
    return merged


def shift(state: List[List[int]], direction: str) -> List[List[int]]:
    """
    Shift all values in the given direction and add a new random tile.
    
    Args:
        state: Current 4x4 grid state (list of lists)
        direction: One of 'left', 'right', 'up', 'down'
    
    Returns:
        New state after shifting and adding a random tile
    """
    # Create a deep copy to avoid modifying the input
    new_state = deepcopy(state)
    
    if direction == 'left':
        for i in range(4):
            new_state[i] = _merge_line(new_state[i])
    
    elif direction == 'right':
        for i in range(4):
            reversed_line = new_state[i][::-1]
            merged = _merge_line(reversed_line)
            new_state[i] = merged[::-1]
    
    elif direction == 'up':
        for j in range(4):
            column = [new_state[i][j] for i in range(4)]
            merged_column = _merge_line(column)
            for i in range(4):
                new_state[i][j] = merged_column[i]
    
    elif direction == 'down':
        for j in range(4):
            column = [new_state[i][j] for i in range(4)]
            reversed_column = column[::-1]
            merged = _merge_line(reversed_column)
            merged_column = merged[::-1]
            for i in range(4):
                new_state[i][j] = merged_column[i]
    
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'left', 'right', 'up', or 'down'")
    
    # Only add a new tile if the state actually changed
    if new_state != state:
        _add_random_tile(new_state)
    
    return new_state


def display(state: List[List[int]]) -> str:
    """
    Display the current game state as a markdown table.
    
    Args:
        state: 4x4 grid state to display
    """
    res = ''
    # Rows
    for row in state:
        res += "| " + " | ".join(f"{val if val else '':^4}" for val in row) + " |\n"
    
    # Score
    score = sum(sum(row) for row in state)
    res += f"\nScore: {score}"

    return res


# Optional helper functions for game logic
def can_move(state: List[List[int]]) -> bool:
    """
    Check if any move is possible from the current state.
    
    Args:
        state: Current 4x4 grid state
    
    Returns:
        True if any move is possible, False otherwise
    """
    for direction in ['left', 'right', 'up', 'down']:
        test_state = shift(state, direction)
        if test_state != state:
            return True
    return False


def is_game_over(state: List[List[int]]) -> bool:
    """
    Check if the game is over (no moves possible).
    
    Args:
        state: Current 4x4 grid state
    
    Returns:
        True if game is over, False otherwise
    """
    return not can_move(state)


def get_score(state: List[List[int]]) -> int:
    """
    Calculate the score (sum of all tiles).
    
    Args:
        state: Current 4x4 grid state
    
    Returns:
        Total score
    """
    return sum(sum(row) for row in state)


if __name__ == "__main__":
    # Example usage
    print("Welcome to 2048!")
    print("Commands: w (up), s (down), a (left), d (right), q (quit)")
    
    current_state = init_grid()
    display(current_state)
    
    while not is_game_over(current_state):
        command = input("\nEnter move: ").lower().strip()
        
        if command == 'q':
            print("Thanks for playing!")
            break
        
        direction_map = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
        
        if command in direction_map:
            new_state = shift(current_state, direction_map[command])
            if new_state != current_state:
                current_state = new_state
                display(current_state)
            else:
                print("Invalid move! Try another direction.")
        else:
            print("Invalid command! Use w/a/s/d to move or q to quit.")
    
    if is_game_over(current_state):
        print("\nGame Over!")
        print(f"Final Score: {get_score(current_state)}")
