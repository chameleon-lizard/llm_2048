"""
LLM-Powered 2048 Game Player
This script uses an LLM to play the 2048 game and logs all moves.
"""

# Set random seed for deterministic gameplay
import random
random.seed(42)

import argparse
import json
import re
import io
import sys
from openai import OpenAI
from game_2048 import init_grid, shift, is_game_over, get_score, display


def get_move_tool_schema():
    """Returns the tool schema for function calling mode."""
    return {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Make a move in the 2048 game by shifting tiles in the specified direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to shift tiles"
                    }
                },
                "required": ["direction"]
            }
        }
    }


def get_llm_move(client, grid_state, messages, model="gpt-4o-mini", max_retries=5, use_function_calling=False):
    """
    Get the next move from the LLM with retry logic.

    Args:
        client: OpenAI client instance
        grid_state: Current game state (4x4 grid)
        messages: List of conversation messages (modified in-place)
        model: OpenAI model to use
        max_retries: Maximum number of attempts to get a valid response
        use_function_calling: If True, use function calling mode; otherwise use text parsing mode

    Returns:
        Tuple of (direction, response_data)
        where response_data is either:
        - text string (text parsing mode)
        - dict with 'reasoning', 'tool_call' keys (function calling mode)
    """
    grid_display = display(grid_state)

    # Prepare prompt based on mode
    if len(messages) == 0:
        if use_function_calling:
            prompt = f"""You are playing the game of 2048. Here are the rules:

2048 is played on a plain 4×4 grid, with numbered tiles that slide in four directions: UP, DOWN, LEFT and RIGHT. The game begins with two tiles already in the grid, having a value of either 2 or 4, and another such tile appears in a random empty space after each turn. Tiles slide as far as possible in the chosen direction until they are stopped by either another tile or the edge of the grid. If two tiles of the same number collide while moving, they will merge into a tile with the total value of the two tiles that collided. The resulting tile cannot merge with another tile again in the same move.

If a move causes three consecutive tiles of the same value to slide together, only the two tiles farthest along the direction of motion will combine. If all four spaces in a row or column are filled with tiles of the same value, a move parallel to that row/column will combine the first two and last two. Your score is sum of all values in the grid.

Analyze the current grid state and choose the best move using the make_move function.

Here is the current grid state:

{grid_display}"""
        else:
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
        messages.append({"role": "user", "content": prompt})
    else:
        prompt = f"""Here is the current grid state:

{grid_display}"""
        if not use_function_calling:
            prompt += "\n\nWhere the values should be shifted next?"
        messages.append({"role": "user", "content": prompt})

    # Retry logic
    last_response = None
    for attempt in range(max_retries):
        try:
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": 0.7
            }
            
            if use_function_calling:
                api_params["tools"] = [get_move_tool_schema()]
                api_params["tool_choice"] = "auto"

            response = client.chat.completions.create(**api_params)
            message = response.choices[0].message

            if use_function_calling:
                # Function calling mode
                reasoning = message.content if message.content else ""
                
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    args = json.loads(tool_call.function.arguments)
                    direction = args["direction"]
                    
                    # Add assistant message with tool call to history
                    messages.append({
                        "role": "assistant",
                        "content": reasoning,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    })
                    
                    # Add tool response (will be added after move validation)
                    # We'll return info needed to add this later
                    response_data = {
                        "reasoning": reasoning,
                        "tool_call": {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": args
                        }
                    }
                    
                    return direction, response_data
                else:
                    print(f"⚠️  Attempt {attempt + 1}/{max_retries}: No tool call in response. Retrying...")
                    last_response = reasoning
            else:
                # Text parsing mode
                full_response = message.content
                last_response = full_response

                match = re.search(r'FINAL_RESPONSE:\s*(UP|DOWN|LEFT|RIGHT)', full_response, re.IGNORECASE)

                if match:
                    direction = match.group(1).lower()
                    messages.append({"role": "assistant", "content": full_response})
                    return direction, full_response
                else:
                    print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Could not parse direction from LLM response. Retrying...")

        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_retries}: API error: {e}. Retrying...")
            last_response = str(e)

    raise ValueError(f"Could not get valid move after {max_retries} attempts. Last response: {last_response}")


def play_game_with_llm(api_key, base_url, log_file="game_log.json", model="gpt-4o-mini", max_moves=1000, max_consecutive_invalid_moves=10, context_window_moves=5, use_function_calling=False):
    """
    Play a full game of 2048 using LLM and log all moves.

    Args:
        api_key: OpenAI API key
        base_url: Base URL for API
        log_file: Path to the JSON log file
        model: OpenAI model to use
        max_moves: Maximum number of moves to prevent infinite loops
        max_consecutive_invalid_moves: Maximum consecutive invalid moves before stopping
        context_window_moves: Number of valid moves to keep in conversation context
        use_function_calling: If True, use function calling mode; otherwise use text parsing mode

    Returns:
        Final score
    """
    client = OpenAI(
            api_key=api_key, base_url=base_url,
            # http_client=__import__('httpx').Client(verify=False)
        )

    current_state = init_grid()
    game_log = []
    move_count = 0
    game_end_reason = "unknown"
    messages = []
    valid_move_history = []
    consecutive_invalid_moves = 0

    # Log initial state with mode
    game_log.append({
        "game_state": [row[:] for row in current_state],
        "action": "INITIAL",
        "current_score": get_score(current_state),
        "mode": "function_calling" if use_function_calling else "text_parsing"
    })

    # Game loop
    while not is_game_over(current_state) and move_count < max_moves:
        try:
            print(f"Move {move_count + 1}, score {get_score(current_state)}, model {model}")

            messages_before = len(messages)

            direction, llm_response = get_llm_move(client, current_state, messages, model, use_function_calling=use_function_calling)

            new_state = shift(current_state, direction)

            # Check if move was valid (state changed)
            if new_state == current_state:
                consecutive_invalid_moves += 1
                print(f"\n⚠️  Invalid move {direction.upper()}! State didn't change. Retrying... ({consecutive_invalid_moves}/{max_consecutive_invalid_moves})")

                # Prepare log entry
                log_entry = {
                    "game_state": [row[:] for row in current_state],
                    "action": direction.upper(),
                    "current_score": get_score(current_state),
                    "invalid_move": True
                }
                
                if use_function_calling:
                    log_entry["llm_reasoning"] = llm_response.get("reasoning", "")
                    log_entry["tool_call"] = llm_response.get("tool_call")
                else:
                    log_entry["llm_reasoning"] = llm_response
                
                game_log.append(log_entry)

                if consecutive_invalid_moves >= max_consecutive_invalid_moves:
                    print(f"\n❌ Too many consecutive invalid moves ({max_consecutive_invalid_moves}). Game stopped.")
                    game_end_reason = f"too_many_invalid_moves_{max_consecutive_invalid_moves}"
                    break

                # Add feedback for invalid move
                if use_function_calling:
                    # Add tool response indicating error
                    messages.append({
                        "role": "tool",
                        "tool_call_id": llm_response["tool_call"]["id"],
                        "content": f"Error: Invalid move. The grid state did not change. No tiles could move or merge in the {direction.upper()} direction. Please choose a different direction."
                    })
                else:
                    feedback = f"That move ({direction.upper()}) was invalid - the grid state did not change. This means no tiles could move or merge in that direction. Please choose a different direction where tiles can actually move."
                    messages.append({"role": "user", "content": feedback})

                continue

            # Valid move
            consecutive_invalid_moves = 0
            current_state = new_state
            move_count += 1

            # Handle valid move history based on mode
            if use_function_calling:
                # In function calling mode, add tool response
                tool_response = {
                    "role": "tool",
                    "tool_call_id": llm_response["tool_call"]["id"],
                    "content": f"Move successful. New score: {get_score(current_state)}"
                }
                messages.append(tool_response)
                
                # Store the full exchange (user, assistant with tool_call, tool response)
                if len(messages) >= 3:
                    user_prompt = messages[messages_before]
                    assistant_response = messages[messages_before + 1]
                    tool_resp = messages[messages_before + 2]
                    
                    valid_move_history.append((user_prompt, assistant_response, tool_resp))
                    
                    if len(valid_move_history) > context_window_moves:
                        valid_move_history = valid_move_history[-context_window_moves:]
                    
                    # Reconstruct messages
                    messages = []
                    for user_msg, assistant_msg, tool_msg in valid_move_history:
                        messages.append(user_msg)
                        messages.append(assistant_msg)
                        messages.append(tool_msg)
            else:
                # Text parsing mode - keep existing logic
                if len(messages) >= 2:
                    user_prompt = messages[messages_before]
                    assistant_response = messages[messages_before + 1]
                    
                    valid_move_history.append((user_prompt, assistant_response))
                    
                    if len(valid_move_history) > context_window_moves:
                        valid_move_history = valid_move_history[-context_window_moves:]
                    
                    messages = []
                    for user_msg, assistant_msg in valid_move_history:
                        messages.append(user_msg)
                        messages.append(assistant_msg)

            # Log the move
            log_entry = {
                "game_state": [row[:] for row in current_state],
                "action": direction.upper(),
                "current_score": get_score(current_state)
            }
            
            if use_function_calling:
                log_entry["llm_reasoning"] = llm_response.get("reasoning", "")
                log_entry["tool_call"] = llm_response.get("tool_call")
            else:
                log_entry["llm_reasoning"] = llm_response
            
            game_log.append(log_entry)

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
    elif game_end_reason.startswith("too_many_invalid_moves"):
        print(f"Game stopped due to too many consecutive invalid moves!")
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
    parser = argparse.ArgumentParser(description='LLM plays 2048 game')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--base_url', type=str, required=True, help='Base URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--context_window', type=int, default=5, help='Number of valid moves to keep in context (default: 5)')
    parser.add_argument('--use_function_calling', action='store_true', help='Use native function calling instead of text parsing')

    args = parser.parse_args()

    # Prepare log file name with function calling marker
    model_short = args.model_name.split('/')[-1]
    fc_suffix = "_fc" if args.use_function_calling else ""
    log_file = f"game_logs/game_log_{model_short}{fc_suffix}.json"

    # Play the game
    play_game_with_llm(
        api_key=args.api_key,
        base_url=args.base_url,
        log_file=log_file,
        model=args.model_name,
        max_moves=10000,
        context_window_moves=args.context_window,
        use_function_calling=args.use_function_calling,
    )
