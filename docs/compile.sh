#!/bin/bash

# Generate LaTex source with Python's docutils
rst2latex theory.rst > theory.tex

# First compiler pass
xelatex theory.tex

# Compile twice to get the correct references
xelatex theory.tex
