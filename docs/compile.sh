#!/bin/bash
#
# usage ./compile <filename>

filename=$1

# Generate LaTex source with Python's docutils
rst2latex.py $filename.rst > build/$filename.tex


cp references.bib build/references.bib

cd build

# First compiler pass
xelatex $filename.tex

bibtex $filename.aux

# Compile twice to get the correct references
xelatex $filename.tex

# because of bib
xelatex $filename.tex