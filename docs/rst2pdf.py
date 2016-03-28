#!/usr/bin/env python
#
import argparse
from pathlib import Path
from subprocess import check_call
from docutils.core import publish_file
from docutils.writers import latex2e
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.body import MathBlock
from docutils.utils.math import pick_math_environment
import sys
from shutil import copy


# Directories
rootdir  = Path(__file__).resolve().parent
builddir = rootdir / 'build'

# Command Line Interface
parser = argparse.ArgumentParser()
parser.add_argument('filename')


class MathDirective(MathBlock):
    option_spec = dict(MathBlock.option_spec,
                       numbered=directives.flag)


directives.register_directive('math', MathDirective)


class LaTeXTranslator(latex2e.LaTeXTranslator):

    def visit_math_block(self, node):
        numbered = 'numbered' in node
        math_env = pick_math_environment(node.astext(), numbered)
        self.visit_math(node, math_env=math_env)


class Writer(latex2e.Writer):

    def __init__(self):
        latex2e.Writer.__init__(self)
        self.translator_class = LaTeXTranslator


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
                 writer=Writer(),
                 settings_overrides={
                     'template': 'template.tex'
                 })
    
    # Copy BibTex references
    copy(str(rootdir / 'references.bib'), str(builddir / 'references.bib'))

    # Compile LaTex to PDF
    # FIXME As long as latexmk is not installable on the working machine, we use
    #       three xelatex calls for building outlines and bibtex
    # check_call(['latexmk', '-xelatex', destination.name], cwd=str(builddir))
    check_call(['xelatex', destination.name], cwd=str(builddir))
    check_call(['xelatex', destination.name], cwd=str(builddir))
    check_call(['xelatex', destination.name], cwd=str(builddir))


if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(rst2pdf(args.filename))
