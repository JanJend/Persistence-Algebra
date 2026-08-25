#!/bin/bash
  
PROGRAM="/home/wsljan/MP-Workspace/Persistence-Algebra/build/resolution"
INPUT_FOLDER="/home/wsljan/MP-Workspace/data/hypoxic_regions/hypoxic2_FoxP3_dim1_200x200_snapped_induced"
OUTPUT_FOLDER="/home/wsljan/MP-Workspace/data/hypoxic_regions/hypoxic2_FoxP3_dim1_200x200_snapped_induced/resolutions"
for file in "$INPUT_FOLDER"/*.scc; do
    if [ -f "$file" ]; then
        "$PROGRAM" "$file" "$OUTPUT_FOLDER/$(basename "$file" .scc)_resolution.scc"
    fi
done