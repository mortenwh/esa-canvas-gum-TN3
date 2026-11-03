# AI_LOGS

## 2026-11-03T12:30:00Z — To-do macro
**Prompt:** Read AGENTS.md; add a `\todo{ID}{details}{date}` macro with a summary list like the acronym package.
**Actions:**
- Created `todo.sty` (inline tag, `.tdl` file, `\listoftodos` longtable).
- Loaded it in `gum4sar.tex`; list added after the acronyms.
- Added `*.tdl` to `.gitignore`; created DOCUMENTATION.md, TRUTHS.md, AI_LOGS.md.
- Tested in a temp document and built `gum4sar.tex` without `!` errors.
**Outcome:** Macro works; no `\todo` entries used in the manuscript yet.
