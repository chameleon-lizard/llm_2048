"""
Plot final scores for each model's 2048 game.
Creates a barplot comparing final scores across all models.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_final_score(log_file):
    """Load a game log JSON file and extract final score and stats."""
    with open(log_file, 'r') as f:
        data = json.load(f)

    # The last entry should contain final stats
    final_entry = data[-1]

    if 'final_score' in final_entry:
        return {
            'final_score': final_entry['final_score'],
            'total_moves': final_entry.get('total_moves', 0),
            'end_reason': final_entry.get('game_end_reason', 'unknown')
        }
    else:
        # Fallback: get the last score from the second-to-last entry
        if len(data) > 1 and 'current_score' in data[-2]:
            return {
                'final_score': data[-2]['current_score'],
                'total_moves': len(data) - 2,
                'end_reason': 'unknown'
            }

    return None


def get_model_name(filename):
    """Extract model name from log filename."""
    return filename.replace('game_log_', '').replace('.json', '')


def plot_final_scores(log_dir='game_logs', output_file='final_scores_barplot.png'):
    """Create a barplot of final scores for all models."""
    log_path = Path(log_dir)

    # Load all final scores
    model_scores = {}

    for log_file in sorted(log_path.glob('game_log_*.json')):
        model_name = get_model_name(log_file.name)
        try:
            stats = load_final_score(log_file)
            if stats:
                model_scores[model_name] = stats
                print(f"{model_name}: Score={stats['final_score']}, Moves={stats['total_moves']}, Reason={stats['end_reason']}")
        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    if not model_scores:
        print("No game logs found!")
        return

    # Sort by final score (descending)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)

    model_names = [name for name, _ in sorted_models]
    scores = [stats['final_score'] for _, stats in sorted_models]
    moves = [stats['total_moves'] for _, stats in sorted_models]

    # Create the barplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Final Scores
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars1 = ax1.bar(range(len(model_names)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Score', fontsize=12, fontweight='bold')
    ax1.set_title('2048 Final Scores by Model', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(score)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot 2: Total Moves
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(model_names)))
    bars2 = ax2.bar(range(len(model_names)), moves, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Moves', fontsize=12, fontweight='bold')
    ax2.set_title('Total Moves by Model', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, move_count) in enumerate(zip(bars2, moves)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(move_count)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBarplot saved to {output_file}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Best Score: {model_names[0]} with {scores[0]}")
    print(f"Worst Score: {model_names[-1]} with {scores[-1]}")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Median Score: {np.median(scores):.1f}")
    print(f"Average Moves: {np.mean(moves):.1f}")
    print(f"Median Moves: {np.median(moves):.1f}")
    print("=" * 60)


def plot_score_vs_moves(log_dir='game_logs', output_file='score_vs_moves.png'):
    """Create a scatter plot of final score vs total moves."""
    log_path = Path(log_dir)

    model_scores = {}

    for log_file in sorted(log_path.glob('game_log_*.json')):
        model_name = get_model_name(log_file.name)
        try:
            stats = load_final_score(log_file)
            if stats:
                model_scores[model_name] = stats
        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    if not model_scores:
        print("No game logs found!")
        return

    model_names = list(model_scores.keys())
    scores = [stats['final_score'] for stats in model_scores.values()]
    moves = [stats['total_moves'] for stats in model_scores.values()]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(moves, scores, s=200, alpha=0.6, c=range(len(model_names)),
                        cmap='viridis', edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, name in enumerate(model_names):
        ax.annotate(name, (moves[i], scores[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('Total Moves', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Score', fontsize=12, fontweight='bold')
    ax.set_title('Final Score vs Total Moves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot 2048 final scores')
    parser.add_argument('--log_dir', type=str, default='game_logs',
                        help='Directory containing game log JSON files')
    parser.add_argument('--output', type=str, default='final_scores_barplot.png',
                        help='Output filename for barplot')
    parser.add_argument('--scatter', action='store_true',
                        help='Also create scatter plot of score vs moves')
    parser.add_argument('--scatter_output', type=str, default='score_vs_moves.png',
                        help='Output filename for scatter plot')

    args = parser.parse_args()

    # Create barplot
    plot_final_scores(args.log_dir, args.output)

    # Create scatter plot if requested
    if args.scatter:
        plot_score_vs_moves(args.log_dir, args.scatter_output)
