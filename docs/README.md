## Documentation

- assignment of tasks
- study thesis

The study thesis are written in [restructuredText][rst] because of its
simplicity and better readability compared with pure LaTex. I use the following
pipline to generate the final PDF:

    restructuredText -> LaTex -> PDF


### Preparation

```bash
# Install neat wrapper 
sudo apt-get install virtualenvwrapper

# Create virtual python environment
mkvirtualenv --python=$(which python3) study-thesis

# Install the required Python packages in the newly created environment
pip install docutils
```

For a fany syntax highlighting in Sublime Text, you can install *RestructuredText Improved* via [Package Control][subl-control].



### Generate PDF from restructured Text

Use the *compile.sh* from the docs folder. For cleaning up everything use the *clear.sh*



### Generate PDF from latex file

```bash
xelatex assignment-of-tasks.tex
```

[rst]: http://docutils.sourceforge.net/rst.html
[subl-control]: https://packagecontrol.io/installation