#!/bin/bash
  
PROGRAM="/home/jan/MP-Workspace/Persistence-Algebra/build/analyse_ind"
FOLDER="${1:-.}"  # Use provided folder or current directory
H_DEGREE="$2"             # Optional: 0 or 1
OUTPUT="$FOLDER/ind_hom_time_analysis_H${H_DEGREE}.md"

# Decide which files to process
if [[ "$H_DEGREE" == "0" || "$H_DEGREE" == "1" ]]; then
    FILES=("$FOLDER"/*"H${H_DEGREE}.sccsum")
else
    FILES=("$FOLDER"/*.sccsum)
fi

# Clear/create the output file
echo "# Computation times for Hom(X,X) for indecomposable X with beta^1(X) > 100" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "Generated on: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Process each file
for file in "${FILES[@]}"; do
    [[ -e "$file" ]] || continue   # avoid literal glob if no match
    echo "## $(basename "$file")" >> "$OUTPUT"
    "$PROGRAM" "$file" true >> "$OUTPUT" 2>&1

done

echo "Results saved to: $OUTPUT"