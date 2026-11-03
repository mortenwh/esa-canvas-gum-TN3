# DOCUMENTATION

LaTeX sources for the GUM for SAR report (TN3, CANVAS). Main file:
`gum4sar.tex`; build with `bash compile.sh` (pdflatex, bibtex, makeglossaries).

## To-do macro (`todo.sty`)

* `\todo{<one-word-ID>}{<details>}{<target-closing-date>}` places a red
  `[TODO: ID]` tag in the text and records ID, details, date and page.
* `\listoftodos` prints the summary table (ID, details, target date, page)
  in a "To-do list" section; it is placed after the acronyms in
  `gum4sar.tex`.
* Entries are written to `gum4sar.tdl` (git-ignored) and read back on the
  next run, so run LaTeX twice (`compile.sh` does).
* Option `\usepackage[final]{todo}` hides the inline tags; the list stays.
* Details are expanded at write time: fragile commands in `<details>`
  should be `\protect`ed.

## Tooling

* `gum_diagram.py` generates uncertainty tree diagram `.tex` files (see
  `CLAUDE.md` for the rules on these files); tests in `test_gum_diagram.py`.

## Status / next steps

* To-do macro implemented and tested; no `\todo` entries in the manuscript yet.
* Open: `guidelines_writing_papers/` (referenced in AGENTS.md) does not exist.
* Open: AGENTS.md says "only edit python scripts"; the user allowed LaTeX
  edits for the to-do macro only (see `TRUTHS.md`).
