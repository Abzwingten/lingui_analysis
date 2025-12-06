#!/bin/bash
# Usage: ./run.sh input/text.txt output/report.html

INPUT_FILE=$1
OUTPUT_FILE=$2

if [ -z "$INPUT_FILE" ]; then
    echo "❌ Error: Input file path is required"
    echo "Usage: $0 <input_file.txt> [output_report.html]"
    exit 1
fi

if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="output/report.html"
fi

# Create directories if they don't exist
mkdir -p $(dirname "$OUTPUT_FILE")

echo "🐳 Building Docker image..."
docker build -t linguistic-analyzer .

echo "🚀 Running analysis on $INPUT_FILE"
docker run --rm \
    -v $(pwd):/app/data \
    -v $(pwd)/$(dirname "$OUTPUT_FILE"):/app/output \
    linguistic-analyzer \
    "/app/data/$INPUT_FILE" \
    -o "/app/output/$(basename "$OUTPUT_FILE")"

echo "✅ Analysis complete! Report saved to $OUTPUT_FILE"
