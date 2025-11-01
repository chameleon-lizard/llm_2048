"""
Create animated GIFs showing game state evolution for each model's 2048 game.
Visualizes the board state at each move.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image
import io


# Color scheme for tiles (similar to the original 2048 game)
TILE_COLORS = {
    0: '#CDC1B4',      # Empty
    2: '#EEE4DA',      # 2
    4: '#EDE0C8',      # 4
    8: '#F2B179',      # 8
    16: '#F59563',     # 16
    32: '#F67C5F',     # 32
    64: '#F65E3B',     # 64
    128: '#EDCF72',    # 128
    256: '#EDCC61',    # 256
    512: '#EDC850',    # 512
    1024: '#EDC53F',   # 1024
    2048: '#EDC22E',   # 2048
    4096: '#3C3A32',   # 4096+
}

# Text colors
TILE_TEXT_COLORS = {
    0: '#CDC1B4',
    2: '#776E65',
    4: '#776E65',
}
DEFAULT_TEXT_COLOR = '#F9F6F2'


def get_tile_color(value):
    """Get the color for a tile value."""
    if value in TILE_COLORS:
        return TILE_COLORS[value]
    else:
        return TILE_COLORS[4096]  # Default for values > 2048


def get_text_color(value):
    """Get the text color for a tile value."""
    if value in TILE_TEXT_COLORS:
        return TILE_TEXT_COLORS[value]
    else:
        return DEFAULT_TEXT_COLOR


def render_game_state(game_state, score, move_num, action, ax):
    """Render a single game state as a matplotlib figure."""
    ax.clear()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw grid
    for i in range(4):
        for j in range(4):
            value = game_state[i][j]

            # Draw tile background
            rect = mpatches.Rectangle((j, 3 - i), 1, 1,
                                     facecolor=get_tile_color(value),
                                     edgecolor='#BBADA0',
                                     linewidth=3)
            ax.add_patch(rect)

            # Draw tile value
            if value != 0:
                text_color = get_text_color(value)
                fontsize = 40 if value < 100 else (32 if value < 1000 else 24)
                ax.text(j + 0.5, 3 - i + 0.5, str(value),
                       ha='center', va='center',
                       fontsize=fontsize, fontweight='bold',
                       color=text_color)

    # Add game info
    info_text = f"Move: {move_num} | Action: {action} | Score: {score}"
    ax.text(2, -0.3, info_text, ha='center', va='top',
           fontsize=14, fontweight='bold', color='#776E65')


def load_game_states(log_file):
    """Load all game states from a log file."""
    with open(log_file, 'r') as f:
        data = json.load(f)

    states = []
    for i, entry in enumerate(data):
        if 'game_state' in entry:
            states.append({
                'state': entry['game_state'],
                'score': entry.get('current_score', 0),
                'action': entry.get('action', 'UNKNOWN'),
                'move_num': i
            })
        elif 'final_score' in entry:
            # Skip final stats entry
            break

    return states


def get_model_name(filename):
    """Extract model name from log filename."""
    return filename.replace('game_log_', '').replace('.json', '')


def create_gif(log_file, output_dir='gifs', fps=2, max_frames=None):
    """Create an animated GIF from a game log."""
    model_name = get_model_name(log_file.name)
    print(f"Creating GIF for {model_name}...")

    try:
        states = load_game_states(log_file)

        if not states:
            print(f"  No states found for {model_name}")
            return

        # Limit frames if specified
        if max_frames and len(states) > max_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
            states = [states[i] for i in indices]

        # Create frames
        frames = []
        fig, ax = plt.subplots(figsize=(6, 6.5))

        for state_info in states:
            render_game_state(
                state_info['state'],
                state_info['score'],
                state_info['move_num'],
                state_info['action'],
                ax
            )

            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='#FAF8EF', edgecolor='none')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        plt.close(fig)

        # Save as GIF
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'game_{model_name}.gif')

        # Calculate duration per frame in milliseconds
        duration = int(1000 / fps)

        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )

        print(f"  ✓ Saved {output_file} ({len(frames)} frames)")

    except Exception as e:
        print(f"  ✗ Error creating GIF for {model_name}: {e}")


def create_all_gifs(log_dir='game_logs', output_dir='gifs', fps=2, max_frames=None):
    """Create GIFs for all game logs."""
    log_path = Path(log_dir)

    log_files = sorted(log_path.glob('game_log_*.json'))

    if not log_files:
        print(f"No game logs found in {log_dir}")
        return

    print(f"Found {len(log_files)} game logs")
    print(f"Creating GIFs at {fps} FPS...")

    if max_frames:
        print(f"Limiting to {max_frames} frames per GIF")

    print()

    for log_file in log_files:
        create_gif(log_file, output_dir, fps, max_frames)

    print(f"\nAll GIFs saved to {output_dir}/")


def create_sample_gif(log_file, output_file, num_frames=20, fps=2):
    """Create a sample GIF with evenly spaced frames."""
    print(f"Creating sample GIF: {output_file}")

    try:
        states = load_game_states(log_file)

        if not states:
            print("  No states found")
            return

        # Sample frames evenly
        indices = np.linspace(0, len(states) - 1, min(num_frames, len(states)), dtype=int)
        sampled_states = [states[i] for i in indices]

        # Create frames
        frames = []
        fig, ax = plt.subplots(figsize=(6, 6.5))

        for state_info in sampled_states:
            render_game_state(
                state_info['state'],
                state_info['score'],
                state_info['move_num'],
                state_info['action'],
                ax
            )

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='#FAF8EF', edgecolor='none')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        plt.close(fig)

        # Save as GIF
        duration = int(1000 / fps)
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )

        print(f"  ✓ Saved {output_file} ({len(frames)} frames)")

    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create animated GIFs of 2048 games')
    parser.add_argument('--log_dir', type=str, default='game_logs',
                        help='Directory containing game log JSON files')
    parser.add_argument('--output_dir', type=str, default='gifs',
                        help='Directory to save GIFs')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second for GIF animation')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames per GIF (samples evenly if exceeded)')
    parser.add_argument('--model', type=str, default=None,
                        help='Create GIF for specific model only (e.g., "claude-sonnet-4.5")')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample GIFs with limited frames (20 frames)')

    args = parser.parse_args()

    if args.model:
        # Create GIF for specific model
        log_file = Path(args.log_dir) / f'game_log_{args.model}.json'
        if log_file.exists():
            if args.sample:
                output_file = os.path.join(args.output_dir, f'game_{args.model}_sample.gif')
                os.makedirs(args.output_dir, exist_ok=True)
                create_sample_gif(log_file, output_file, num_frames=20, fps=args.fps)
            else:
                create_gif(log_file, args.output_dir, args.fps, args.max_frames)
        else:
            print(f"Log file not found: {log_file}")
    else:
        # Create GIFs for all models
        max_frames = 20 if args.sample else args.max_frames
        create_all_gifs(args.log_dir, args.output_dir, args.fps, max_frames)
