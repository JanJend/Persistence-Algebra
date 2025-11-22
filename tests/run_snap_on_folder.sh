#!/bin/bash

PROGRAM="/home/wsljan/MP-Workspace/Persistence-Algebra/build/snap_grid"
INPUT_FOLDER="/path/to/your/folder"  # Change this

for file in "$INPUT_FOLDER"/*; do
    if [ -f "$file" ]; then
        for i in {3..15}; do
            "$PROGRAM" "$file" "$i"
        done
    fi
done