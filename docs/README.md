## Documentation

- assignment of tasks
- study thesis

The study thesis are written in [restructuredText][rst] because of its
simplicity and better readability compared with pure LaTex. I use the following
pipeline to generate the final PDF:

    restructuredText -> LaTex -> PDF


### Preparation

```bash
# Install neat wrapper 
sudo apt-get install virtualenvwrapper

# Create virtual python environment
mkvirtualenv --python=$(which python3) study-thesis

# Install the required Python packages in the newly created environment
pip install docutils watchdog
```

For a fancy syntax highlighting in Sublime Text, you can install *RestructuredText Improved* via [Package Control][subl-control].


### Generate PDF from restructured Text

```bash
# activate virtual environment
workon study-thesis

# compile study-thesis.rst
cd docs/
./rst2pdf.py study-thesis.rst

# If you want to recompile our the documentation each time a file
# is modified, use the "watch" script
./watch
```

A template.tex is used for compiling the reST files because it allows defining the latex preamble the way it fits my needs.

The PDF file can be found in docs/build. For cleaning up everything use the *clear.sh*


### Generate PDF from latex file

```bash
xelatex -output-directory=../build/ assignment-of-tasks.tex 
```

[rst]: http://docutils.sourceforge.net/rst.html
[subl-control]: https://packagecontrol.io/installation
