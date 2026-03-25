#!/usr/bin/env bash
pdflatex gum4sar.tex
pdflatex gum4sar.tex # Get acronyms
bibtex gum4sar
makeglossaries gum4sar
pdflatex gum4sar.tex
pdflatex gum4sar.tex # Get references and glossary
