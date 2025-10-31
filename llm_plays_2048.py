"""
LLM-Powered 2048 Game Player
This script uses an LLM to play the 2048 game and logs all moves.
"""

import argparse
import json
import re
import io
import sys
from openai import OpenAI
from game_2048 import init_grid, shift, is_game_over, get_score, display


def get_llm_move(client, grid_state, model="gpt-4o-mini"):
    """
    Get the next move from the LLM.
    
    Args:
        client: OpenAI client instance
        grid_state: Current game state (4x4 grid)
        model: OpenAI model to use
    
    Returns:
        Tuple of (direction, full_response)
    """
    # Use the display function to format the grid state
    grid_display = display(grid_state)
    
    prompt = f"""You are playing the game of 2048. Here are the rules:

2048 is played on a plain 4×4 grid, with numbered tiles that slide in four directions: UP, DOWN, LEFT and RIGHT. The game begins with two tiles already in the grid, having a value of either 2 or 4, and another such tile appears in a random empty space after each turn. Tiles slide as far as possible in the chosen direction until they are stopped by either another tile or the edge of the grid. If two tiles of the same number collide while moving, they will merge into a tile with the total value of the two tiles that collided. The resulting tile cannot merge with another tile again in the same move.

If a move causes three consecutive tiles of the same value to slide together, only the two tiles farthest along the direction of motion will combine. If all four spaces in a row or column are filled with tiles of the same value, a move parallel to that row/column will combine the first two and last two. Your score is sum of all values in the grid.

Your task is to select a direction of the shift. You may think for as long as you like, but then you need to say on a separate from your reasoning line:

FINAL_RESPONSE: <direction of the shift in uppercase>

Example of the response:

I think, I should shift everything to the right.

FINAL_RESPONSE: RIGHT

Here is the current grid state:

{grid_display}

Where the values should be shifted next?"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    full_response = response.choices[0].message.content
    
    # Extract the direction from FINAL_RESPONSE
    match = re.search(r'FINAL_RESPONSE:\s*(UP|DOWN|LEFT|RIGHT)', full_response, re.IGNORECASE)
    
    if match:
        direction = match.group(1).upper()
        # Convert to lowercase for the shift function
        direction_map = {
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right'
        }
        return direction_map[direction], full_response
    else:
        raise ValueError(f"Could not parse direction from LLM response: {full_response}")


def play_game_with_llm(api_key, base_url, log_file="game_log.json", model="gpt-4o-mini", max_moves=1000):
    """
    Play a full game of 2048 using LLM and log all moves.
    
    Args:
        api_key: OpenAI API key
        log_file: Path to the JSON log file
        model: OpenAI model to use
        max_moves: Maximum number of moves to prevent infinite loops
    
    Returns:
        Final score
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Initialize game
    current_state = init_grid()
    game_log = []
    move_count = 0
    game_end_reason = "unknown"
    
    # Log initial state
    game_log.append({
        "game_state": [row[:] for row in current_state],
        "action": "INITIAL",
        "current_score": get_score(current_state)
    })
    
    # Game loop
    while not is_game_over(current_state) and move_count < max_moves:
        try:
            print(f"Move {move_count + 1}, score {get_score(current_state)}, model {model}")
            
            direction, llm_response = get_llm_move(client, current_state, model)
            
            # Apply move
            new_state = shift(current_state, direction)
            
            # Check if move was valid (state changed)
            if new_state == current_state:
                print("\n⚠️  Invalid move! State didn't change. Game stopped.")
                game_end_reason = "invalid_move"
                break
            
            current_state = new_state
            move_count += 1
            
            # Log the move
            game_log.append({
                "game_state": [row[:] for row in current_state],
                "action": direction.upper(),
                "current_score": get_score(current_state),
                "llm_reasoning": llm_response
            })
            
            # Save log after each move
            with open(log_file, 'w') as f:
                json.dump(game_log, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            game_end_reason = f"error: {str(e)}"
            break
    
    # Determine game end reason if not already set
    if game_end_reason == "unknown":
        if move_count >= max_moves:
            game_end_reason = "max_moves_reached"
        elif is_game_over(current_state):
            game_end_reason = "no_moves_available"
    
    # Game over
    print("\n" + "=" * 50)
    if move_count >= max_moves:
        print("Maximum moves reached!")
    elif game_end_reason == "invalid_move":
        print("Game stopped due to invalid move!")
    elif game_end_reason.startswith("error:"):
        print(f"Game stopped due to error!")
    else:
        print("Game Over!")
    print("=" * 50)
    
    final_score = get_score(current_state)
    print(f"Final Score: {final_score}")
    print(f"Total Moves: {move_count}")
    print(f"Game End Reason: {game_end_reason}")
    print(f"Game log saved to: {log_file}")
    
    # Add final statistics to the log
    game_log.append({
        "final_score": final_score,
        "game_end_reason": game_end_reason,
        "total_moves": move_count
    })
    
    # Save final log
    with open(log_file, 'w') as f:
        json.dump(game_log, f, indent=2)
    
    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--base_url', type=str, required=True, help='Base URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')

    args = parser.parse_args()
    
    # Play the game
    play_game_with_llm(
        api_key=args.api_key,
        base_url=args.base_url,
        log_file=f"game_logs/game_log_{args.model_name.split('/')[1]}.json",
        model=args.model_name,
        max_moves=10000,
    )
