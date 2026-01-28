#!/bin/bash
  
PROGRAM="/home/wsljan/MP-Workspace/Persistence-Algebra/build/analyse_ind"
FOLDER="${1:-.}"  # Use provided folder or current directory
OUTPUT="$FOLDER/ind_analysis.md"

# Clear/create the output file
echo "# End-algebra dimensions" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "Generated on: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Process each file
for file in "$FOLDER"/*.sccsum; do
    
    echo "## $(basename "$file")" >> "$OUTPUT"
    "$PROGRAM" "$file" >> "$OUTPUT" 2>&1

done

echo "Results saved to: $OUTPUT"