"""
Plot scores per turn for each model's 2048 game.
Creates line plots showing score progression over moves.
"""

import json
import os
import matplotlib.pyplot as plt
from pathlib import Path


def load_game_log(log_file):
    """Load a game log JSON file and extract scores per turn."""
    with open(log_file, 'r') as f:
        data = json.load(f)

    # Extract scores (skip the last entry which is final stats)
    scores = []
    moves = []

    for i, entry in enumerate(data):
        if 'current_score' in entry:
            scores.append(entry['current_score'])
            moves.append(i)
        elif 'final_score' in entry:
            # This is the final stats entry
            break

    return moves, scores


def get_model_name(filename):
    """Extract model name from log filename."""
    # filename format: game_log_<model_name>.json
    return filename.replace('game_log_', '').replace('.json', '')


def plot_all_scores(log_dir='game_logs', output_file='scores_per_turn.png'):
    """Plot scores per turn for all models."""
    log_path = Path(log_dir)

    # Load all game logs
    model_data = {}

    for log_file in sorted(log_path.glob('game_log_*.json')):
        model_name = get_model_name(log_file.name)
        try:
            moves, scores = load_game_log(log_file)
            model_data[model_name] = (moves, scores)
            print(f"Loaded {model_name}: {len(moves)} moves, final score {scores[-1] if scores else 0}")
        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    if not model_data:
        print("No game logs found!")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each model
    for model_name, (moves, scores) in sorted(model_data.items()):
        ax.plot(moves, scores, marker='o', markersize=2, linewidth=1.5, label=model_name, alpha=0.8)

    ax.set_xlabel('Move Number', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('2048 Game Score Progression by Model', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()


def plot_individual_scores(log_dir='game_logs', output_dir='plots'):
    """Create individual plots for each model."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(log_dir)

    for log_file in sorted(log_path.glob('game_log_*.json')):
        model_name = get_model_name(log_file.name)
        try:
            moves, scores = load_game_log(log_file)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(moves, scores, marker='o', markersize=3, linewidth=2, color='#2E86AB')
            ax.fill_between(moves, scores, alpha=0.3, color='#2E86AB')

            ax.set_xlabel('Move Number', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Score Progression: {model_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'score_progression_{model_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved {output_file}")
        except Exception as e:
            print(f"Error plotting {log_file}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot 2048 game scores per turn')
    parser.add_argument('--log_dir', type=str, default='game_logs',
                        help='Directory containing game log JSON files')
    parser.add_argument('--output', type=str, default='scores_per_turn.png',
                        help='Output filename for combined plot')
    parser.add_argument('--individual', action='store_true',
                        help='Also create individual plots for each model')
    parser.add_argument('--individual_dir', type=str, default='plots',
                        help='Directory for individual plots')

    args = parser.parse_args()

    # Create combined plot
    plot_all_scores(args.log_dir, args.output)

    # Create individual plots if requested
    if args.individual:
        plot_individual_scores(args.log_dir, args.individual_dir)
