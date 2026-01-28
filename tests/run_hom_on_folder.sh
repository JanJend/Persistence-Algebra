#!/bin/sh

FOLDER="${1:-.}"  # Use provided folder or current directory
PROGRAM="/home/wsljan/MP-Workspace/Persistence-Algebra/build/hom"
OUTPUT="$FOLDER/experiments_type1.md"

# Clear/create the output file
echo "# Experiment Results" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "Generated on: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Process each file
for file in "$FOLDER"/*.scc "$FOLDER"/*.firep; do
    [ -e "$file" ] || continue
    
    echo "## $(basename "$file")" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    
    "$PROGRAM" "$file" "$file" -1 1 >> "$OUTPUT" 2>&1
    
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Results saved to: $OUTPUT"