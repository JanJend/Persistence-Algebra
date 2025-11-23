#!/bin/bash
  
PROGRAM="/home/wsljan/MP-Workspace/Persistence-Algebra/build/snap_grid"
INPUT_FOLDER="/home/wsljan/MP-Workspace/data/hypoxic_regions/hypoxic2_FoxP3_dim1_200x200_snapped_induced"
  
for file in "$INPUT_FOLDER"/*.scc; do
    if [ -f "$file" ]; then
        i=2
        while [ $i -le 32 ]; do
            "$PROGRAM" "$file" "$i"
            i=$((i * 2))
        done
    fi
done