# AI_LOGS

## 2026-11-03T12:30:00Z — To-do macro
**Prompt:** Read AGENTS.md; add a `\todo{ID}{details}{date}` macro with a summary list like the acronym package.
**Actions:**
- Created `todo.sty` (inline tag, `.tdl` file, `\listoftodos` longtable).
- Loaded it in `gum4sar.tex`; list added after the acronyms.
- Added `*.tdl` to `.gitignore`; created DOCUMENTATION.md, TRUTHS.md, AI_LOGS.md.
- Tested in a temp document and built `gum4sar.tex` without `!` errors.
**Outcome:** Macro works; no `\todo` entries used in the manuscript yet.

## 2026-11-03T12:50:00Z — To-do macro draft-only
**Prompt:** Make the todo package print todos only in draft mode (IDs in margin, summary list).
**Actions:**
- `todo.sty`: new `draft`/`final` options (default final); in draft, ID goes in the margin and `\listoftodos` prints; in final both are empty.
- `gum4sar.tex`: now loads `\usepackage[draft]{todo}`.
- Tested both modes in a temp document; full `compile.sh` build has no `!` errors.
**Outcome:** Done. Package option chosen over the class `draft` option to keep figures visible; obeyDraft is not installed.

## 2026-11-03T13:30:00Z — Automatic clickable to-do IDs
**Prompt:** See AGENTS.md (new task: dynamic TBD<number> IDs, hyperref links to the summary table).
**Actions:**
- `todo.sty`: `\todo{details}{date}`, counter-based `TBD<n>` IDs, margin ID links to table row, table page links back.
- `level1.tex`: migrated the one existing `\todo` call; `gum4sar.tex` comment updated.
- Built with `compile.sh` (no `!` errors); verified link destinations in an uncompressed PDF; final mode tested in a temp doc.
**Outcome:** Done; docs updated.

## 2026-11-03T13:20:00Z — Table ID as back-link
**Prompt:** Make the ID (not the page number) in the to-do table the clickable link.
**Actions:**
- `todo.sty`: table ID now links to the margin note; page number is plain; row anchor kept.
- DOCUMENTATION.md updated; rebuilt with `compile.sh`.
**Outcome:** Done.
