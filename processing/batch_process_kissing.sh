#!/bin/bash
# Batch processing script for kissing artifact removal
# Uses all available CPU cores for maximum speed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Batch Kissing Artifact Removal ===${NC}"
echo "This script will process vessel segmentation files in parallel"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "kissing" ]]; then
    echo -e "${YELLOW}Warning: 'kissing' conda environment not activated${NC}"
    echo "Please run: conda activate kissing"
    exit 1
fi

# Default values
INPUT_DIR="/home/jiwoo/urp/data/uan/original"
OUTPUT_DIR="./processed_no_kissing"
PATTERN="*_MRA1_seg.nii.gz"
WORKERS=$(nproc)  # Use all available cores
LOG_FILE="batch_processing_$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -j|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --input DIR      Input directory (default: /home/jiwoo/urp/data/uan/original)"
            echo "  -o, --output DIR     Output directory (default: ./processed_no_kissing)"
            echo "  -p, --pattern PAT    File pattern (default: *_MRA1_seg.nii.gz)"
            echo "  -j, --workers N      Number of workers (default: all CPUs)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Show configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  File pattern: $PATTERN"
echo "  Workers: $WORKERS"
echo "  Log file: $LOG_FILE"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Count files
FILE_COUNT=$(find "$INPUT_DIR" -name "$PATTERN" -type f | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No files found matching pattern: $PATTERN${NC}"
    exit 1
fi

echo -e "${GREEN}Found $FILE_COUNT files to process${NC}"

# Estimate time
ESTIMATED_TIME=$((FILE_COUNT * 26 / WORKERS))  # ~26 seconds per file
echo -e "Estimated processing time: ${YELLOW}$ESTIMATED_TIME seconds${NC} (~$((ESTIMATED_TIME / 60)) minutes)"
echo ""

# Ask for confirmation
read -p "Proceed with processing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Processing cancelled"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start processing
echo ""
echo -e "${GREEN}Starting parallel processing...${NC}"
START_TIME=$(date +%s)

# Run the processing
python fast_parallel_kissing_removal.py \
    "$INPUT_DIR/$PATTERN" \
    "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --log-file "$LOG_FILE"

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Show results
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Processing completed successfully!${NC}"
else
    echo -e "${RED}✗ Processing completed with errors${NC}"
fi

echo "Total time: $DURATION seconds (~$((DURATION / 60)) minutes)"
echo "Log file: $LOG_FILE"

# Show summary from log
echo ""
echo "Summary:"
tail -n 10 "$LOG_FILE" | grep -E "(Total files:|Successful:|Failed:|Average|processing rate:)"

# Count output files
OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*.nii.gz" -type f | wc -l)
echo ""
echo "Output files created: $OUTPUT_COUNT"

exit $EXIT_CODE 