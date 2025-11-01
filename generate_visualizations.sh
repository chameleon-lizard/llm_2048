#!/bin/bash

# Generate all visualizations for 2048 game logs
# Usage: ./generate_visualizations.sh <input_dir> <output_dir>

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_dir  - Directory containing game log JSON files"
    echo "  output_dir - Directory where all visualizations will be saved"
    echo ""
    echo "Example:"
    echo "  $0 game_logs visualizations"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Check if there are any game log files
LOG_COUNT=$(find "$INPUT_DIR" -name "game_log_*.json" | wc -l)
if [ "$LOG_COUNT" -eq 0 ]; then
    echo "Error: No game log files (game_log_*.json) found in '$INPUT_DIR'"
    exit 1
fi

echo "=========================================="
echo "2048 Visualization Generator"
echo "=========================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Game logs found:  $LOG_COUNT"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create subdirectories for organization
mkdir -p "$OUTPUT_DIR/individual_plots"
mkdir -p "$OUTPUT_DIR/gifs"

echo "[1/3] Generating score progression plots..."
python3 plot_scores_per_turn.py \
    --log_dir "$INPUT_DIR" \
    --output "$OUTPUT_DIR/scores_per_turn.png" \
    --individual \
    --individual_dir "$OUTPUT_DIR/individual_plots"

if [ $? -eq 0 ]; then
    echo "✓ Score progression plots completed"
else
    echo "✗ Error generating score progression plots"
    exit 1
fi

echo ""
echo "[2/3] Generating final scores barplots..."
python3 plot_final_scores.py \
    --log_dir "$INPUT_DIR" \
    --output "$OUTPUT_DIR/final_scores_barplot.png" \
    --scatter \
    --scatter_output "$OUTPUT_DIR/score_vs_moves.png"

if [ $? -eq 0 ]; then
    echo "✓ Final scores barplots completed"
else
    echo "✗ Error generating final scores barplots"
    exit 1
fi

echo ""
echo "[3/3] Generating animated GIFs..."
echo "Note: This may take a while for games with many moves..."
python3 create_game_gifs.py \
    --log_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/gifs" \
    --fps 5 \
    --max_frames 100

if [ $? -eq 0 ]; then
    echo "✓ Animated GIFs completed"
else
    echo "✗ Error generating animated GIFs"
    exit 1
fi

echo ""
echo "=========================================="
echo "All visualizations generated successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/scores_per_turn.png"
echo "  - $OUTPUT_DIR/final_scores_barplot.png"
echo "  - $OUTPUT_DIR/score_vs_moves.png"
echo "  - $OUTPUT_DIR/individual_plots/*.png"
echo "  - $OUTPUT_DIR/gifs/*.gif"
echo ""
echo "Done!"
