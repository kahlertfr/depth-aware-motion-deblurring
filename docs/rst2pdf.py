#!/usr/bin/env python
#
import argparse
from pathlib import Path
from subprocess import check_call
from docutils.core import publish_file
import sys
from shutil import copy


# Directories
rootdir  = Path(__file__).resolve().parent
builddir = rootdir / 'build'

# Command Line Interface
parser = argparse.ArgumentParser()
parser.add_argument('filename')


def rst2pdf(filename):
    """Creates a PDF from a file written in restructuredText by using"""
    filename = Path(filename)

    # Ensure that the requested file exists
    if not filename.exists():
        print("Error: File %s does not exist" % filename, file=sys.stderr)
        return 1

    # Create build directory
    if not builddir.exists():
        builddir.mkdir(mode=0o755)

    destination = builddir / "{filename.stem}.tex".format(filename=filename)

    # Compile restructuredText to LaTex
    publish_file(source_path=str(filename),
                 destination_path=str(destination),
                 writer_name='latex',
                 settings_overrides={
                     'template': 'template.tex'
                 })
    
    # Copy BibTex references
    copy(str(rootdir / 'references.bib'), str(builddir / 'references.bib'))

    # Compile LaTex to PDF
    check_call(['latexmk', '-xelatex', destination.name], cwd=str(builddir))


if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(rst2pdf(args.filename))
