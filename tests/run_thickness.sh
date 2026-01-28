#!/bin/bash
  
PROGRAM="/home/jan/MP-Workspace/Persistence-Algebra/build/thickness"
FOLDER="${1:-.}"  # Use provided folder or current directory
OUTPUT="$FOLDER/a_thickness_analysis.md"

# Clear/create the output file
echo "# Layer thickness of $FOLDER" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "Generated on: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Process each file
for file in "$FOLDER"/*.sccsum; do
    
    echo "## $(basename "$file")" >> "$OUTPUT"
    "$PROGRAM" "$file" >> "$OUTPUT" 2>&1

done

echo "Results saved to: $OUTPUT"