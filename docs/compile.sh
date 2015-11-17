#!/bin/bash
#
# usage ./compile <filename>

filename=$1

# Generate LaTex source with Python's docutils
rst2latex.py $filename.rst > build/$filename.tex

cd build

# First compiler pass
xelatex $filename.tex

# Compile twice to get the correct references
xelatex $filename.tex
