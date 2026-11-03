# DOCUMENTATION

LaTeX sources for the GUM for SAR report (TN3, CANVAS). Main file:
`gum4sar.tex`; build with `bash compile.sh` (pdflatex, bibtex, makeglossaries).

## To-do macro (`todo.sty`)

* Draft mode only: `\usepackage[draft]{todo}` (as in `gum4sar.tex`) enables
  todos; switch to `[final]` (the package default) and `\todo` and
  `\listoftodos` print nothing and no `.tdl` file is written.
* The package option is used instead of the class option `draft`, which would
  also turn all graphicx figures into empty boxes.
* `\todo{<one-word-ID>}{<details>}{<target-closing-date>}` (draft) puts a red
  `TODO: ID` note in the margin and records ID, details, date and page.
* `\listoftodos` (draft) prints the summary table (ID, details, target date,
  page) in a "To-do list" section; it is placed after the acronyms in
  `gum4sar.tex`.
* Entries are written to `gum4sar.tdl` (git-ignored) and read back on the
  next run, so run LaTeX twice (`compile.sh` does).
* Details are expanded at write time: fragile commands in `<details>`
  should be `\protect`ed.

## Tooling

* `gum_diagram.py` generates uncertainty tree diagram `.tex` files (see
  `CLAUDE.md` for the rules on these files); tests in `test_gum_diagram.py`.

## Status / next steps

* To-do macro (draft-only, margin IDs) implemented and tested; no `\todo` entries in the manuscript yet.
* Open: `guidelines_writing_papers/` (referenced in AGENTS.md) does not exist.
* Open: AGENTS.md says "only edit python scripts"; the user allowed LaTeX
  edits for the to-do macro only (see `TRUTHS.md`).
