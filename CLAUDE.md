# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM benchmark project that tests different language models by having them play the 2048 game. The project consists of a stateless 2048 game implementation and an LLM player that uses various models via OpenRouter to play the game autonomously.

## Core Architecture

### Two-Module Design

1. **game_2048.py**: Pure functional 2048 game engine
   - Stateless implementation using pure functions (no classes)
   - Core functions: `init_grid()`, `shift(state, direction)`, `is_game_over(state)`, `get_score(state)`, `display(state)`
   - Grid state is a 4x4 list of lists (List[List[int]])
   - All game logic uses immutable operations (deepcopy for state transitions)
   - Can be run standalone for human play with WASD controls

2. **llm_plays_2048.py**: LLM game player with logging
   - Uses OpenAI-compatible API (via OpenRouter) to get move decisions from LLMs
   - Implements retry logic (max 5 attempts) for parsing LLM responses
   - Expects LLM to respond with "FINAL_RESPONSE: <DIRECTION>" format
   - Logs all game states, moves, and LLM reasoning to JSON files
   - Deterministic gameplay (random.seed(42) set at module level)
   - Stops on invalid moves (when state doesn't change)

### Data Flow

```
init_grid() -> LLM analyzes grid -> shift(state, direction) -> new state + random tile -> repeat until game over
```

All moves and states are logged to `game_logs/game_log_<model_name>.json` with:
- Game state snapshots
- LLM reasoning
- Actions taken
- Current score
- Final statistics (total moves, end reason)

## Running the Code

### Play 2048 manually
```bash
python3 game_2048.py
```
Use W (up), A (left), S (down), D (right), Q (quit)

### Run LLM player for a single model
```bash
python3 llm_plays_2048.py \
  --model_name <model_name> \
  --base_url <api_base_url> \
  --api_key <api_key>
```

### Run benchmark across multiple models
```bash
bash launch.sh
```
This runs multiple LLM models in parallel as background processes. Edit launch.sh to modify the model list or API credentials.

## Important Implementation Details

### Determinism
The random seed is set to 42 at the module level in llm_plays_2048.py (line 8) to ensure reproducible game sequences. This means the same initial grid and random tile placements occur for all runs.

### Game State Representation
- Empty cells are represented as 0
- Grid coordinates use standard [row][column] indexing
- The score is the sum of all tile values (not the traditional 2048 scoring)

### Move Validation
The shift() function returns the original state unchanged if a move is invalid (no tiles can move/merge). The LLM player detects this by comparing states and stops the game on invalid moves.

### Retry Logic
LLM responses are retried up to 5 times if:
- The response doesn't contain the required "FINAL_RESPONSE: <DIRECTION>" format
- An API error occurs

After 5 failed attempts, the game terminates with an error.

### Game End Conditions
Games end when:
1. No valid moves remain (traditional game over)
2. Max moves reached (10,000 by default)
3. LLM makes an invalid move
4. LLM response parsing fails after retries
5. API error occurs

## File Locations

- Game logs: `game_logs/game_log_<model_name>.json`
- Model name in filename uses the part after "/" in the full model path (e.g., "anthropic/claude-sonnet-4.5" -> "claude-sonnet-4.5")
