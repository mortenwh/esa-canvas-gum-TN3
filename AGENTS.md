# Guidelines

See @gum4sar.pdf and @CANVAS_proposal.pdf for the project context. This
repository contains the latex and related documents for the GUM for SAR report
(TN3).

## Background information

* All documents in the current folder

# Tasks

* The current version of the document does not compile. I am using `pdflatex
  gum4sar.tex`, and there seems to be an issue with the todo package. Please
  review and suggest the way forward.

# Misc

* make sure to follow all these instructions, keep track and up to date all
  necessary .md files, etc; create missing .md files if necessary; each time
  you are finished with a large task, take a good check and update all files
  including .md files as necessary
* If you see that rules have not been followed in any of the documents, suggest
  to update the documents according to the rules.
* generally use your knowledge of latex and programming to apply best practices
  on all aspects of the coding
* never run "one off complicated commands": write to file, then run the file
* if logging to a file, make sure that logging happens live by using for
  example `--no-capture-output` with `conda run`, or similar with other
  commands
* follow safe development practices; dont leak any sensitive data, make sure
  there are no malicious or vulnerable code, make sure the code is safe to run
  and does what it advertises and what the user expects
* at the end of each update to the code, go through the files and check that
  the documentation and readme are up to date; make sure all important aspects
  are documented in the DOCUMENTATION.md file
* when writing, check that all guidelines present here are followed
* if you find a better way to do things, or you have an idea of possible
  improvement or something better to do (better method, better algorithm,
  better data structure to use; something more to test and try to
  see if it can give a better result), discuss this with the user
* the user is typically a PhD level expert on the topic - dont hesitate to
  discuss, brainstorm, exchange etc with the user on hard questions
* git and version control are managed by the user; do not make commits, create
  branches, or perform any git operations unless explicitly asked
* do not store raw prompts/data containing personal or confidential
  information; redact or summarize minimally, both in the README,
  DOCUMENTATION, AGENT, or any other file: do not leak credentials or passwords
  or secrets
* if you notice anything that may look like a leaked secret or a security risk,
  let the user know immediately
* if writing more languages in the project as a whole, also apply best
  practices relevant to this language; for example:
  * if writing and running bash scripts, use `shellcheck` (either already
    installed or `mamba` installable from conda-forge) and run it on the
    script, and use best practices (headers to fail on first error etc)
* every time you have considered an answer or solution, consider alternatives
  and / or the opposite. Does the answer still make sense or should it be
  reconsidered? Can more view-points be taken, or alternative solutions be
  suggested with the human?
* when starting a new session, make sure to read the whole project and to get
  full understanding about it - this file, all relevant .md files,
  AI_REASONING.md if it exists, code if it exists, etc. 
* also feel free to suggest meta improvements to this AGENTS.md file - for
  example if more automatic checks or tooling have become available, feel free
  to suggest adding these here, or if you see possible better recommendations
  based on actual problems you have to solve, you can suggest feedback to this
  file. Meta-improvement is good!
* if using sub-agents or similar, do not use another and especially not a more
  expensive model than the current one without asking the user. For example, if
  the present model is a sonnet model, creating a sonnet sub-agent or similar is
  ok. But it is not ok to create an opus or fable sub-agent. By default, use the
  same model for new sub-agents or similar that you are, and if you want to use a
  different one, ask the user first if this is ok.
* do not "bullshit reason" by yourself for things you can write a small code
  snippet to answer - instead, write the formally correct code snippet (for
  example in python; write it to disk, never run "on-the-fly", to keep track of
  it) to get the guaranteed correct answer. For example, solve equations with
  sympy to make sure the answer is guaranteed to be correct instead of
  "reasoning" through the solution steps. This applies to many more cases than
  just solving an equation.

# Guidelines for the writing process

* Before any change to files, present a short plan in chat and wait for the
  user's explicit approval. This applies to every task, including
  documentation-only edits. The plan lists: goal, files to touch, steps with
  a verification for each, and anything destructive (e.g., large deletions or
  rewrites). Do not edit, create or delete files until approved.
  Read-only investigation is allowed before the plan.
  If the CLI is in plan mode, use it; otherwise present the plan with the
  ask_user tool.
* Do not write or code anything before the plan is approved

# Guidelines for the LaTex writing

Before anything, read this file and all relevant .md files - AI_REFLECTIONS.md,
README.md, TRUTHS.md, AI_LOGS.md to understand what this is about. If some do
not exist, skip. Also read the guidelines_writing_papers/README.md .

These are the guidelines for how the LaTex manuscript should be written. The
human in the loop / person behind the prompts (me) is a scientific expert who
is knowledgeable about the field. If you disagree with the human (me) about the
science/content, you should start discussing with the human until you are sure
we understand each other. Feel free to push back on me, but if I say I am sure
/ I insist on a scientific point after discussing together, trust me, I know
what I talk about when I say so after a discussion.

If looking at a manuscript for which there are both bullet points in the .md
file and some latex main.tex files, be aware that the user may have edited the
latex file directly by hand. So, if you are asked to update the manuscript
either following bullet point updates, or on-the-fly, make sure to preserve
both the content of the bullet points and of the latex file and to respect and
extend both. If you find some important point are present in the latex .tex
file but not in the .md bullet points file, you can update the bullet points
file as relevant. Generally, try to keep all .md files up to date when you
perform changes.

* always use the .tex files as authoritative source. Ask the user in case of
  conflicts or if something is unclear, then update the .md files accordingly.
* always write out acronyms at first time use. Use the acronyms package to
  summarize the acronyms in a section before the introduction.

## Guidelines for the LaTex writing

* Have equations on their own line follow the LaTex pattern (this is just to
  illustrate with an example; see the `begin` kind used and the `\label`):

```latex
\begin{equation}
    \omega^2 = \frac{gk + Bk^5}{\coth(kH)}.
    \label{eqn:reduced_disprel}
\end{equation}
```

* Have equations or formula within a line be part of the text and inserted
  through `$` at the start and end (this is just to illustrate with an
  example):

```latex
The phase shift of the cross-spectral density between sensors 1 and 2, $\Delta
\phi_{1, 2}(f) = \mathrm{phase}(P_{1,2}(f))$, is calculated at the frequencies
for which coherence about the threshold is observed.
```

* Have figures follow the pattern (notice the `begin` kind used, `label`,
  `caption`, and `includegraphics` patterns, this is just to illustrate with an
  example):

```latex
\begin{figure}
    \centering
    \includegraphics[width=0.49\linewidth]{figs/t15_PSD_Tempelfjorden_2015_segment_2.png}
    \includegraphics[width=0.49\linewidth]{figs/t15_PSD_Tempelfjorden_2015_segment_16.png}
    \caption{PSDs obtained from the 3 VN100 IMUs that are part of the T15 dataset. Left: PSDs from the segment number 2. A clear high-frequency peak is observed in IMUs VN1002 and VN1007 (farther in the ice), but not in VN1005 (closest to the open water). The high-frequency peak is relatively broad. Right: PSDs from the segment number 16. No high-frequency peak is observed.}
    \label{fig:t15_psd}
\end{figure}
```

* Use `citet` and `citep` as relevant (this is just to illustrate with an example):

```latex
Bound waves, which canonical example is the higher-order components of the
Stokes solution \citep{lamb1924hydrodynamics}, are perturbations to the
first-order wave solution that are present as the higher-order consequence of
nonlinear terms.
```

```latex
For example, \citet{zhao2017gap} suggests that arrays of adjacent floating
bodies (in our case, ice floes) can lead to nonlinear resonance and create
high-frequency peaks in PSDs.
```

* All references should come from the `.bib` file. The `.bib` file should be
  exactly correct - only use exact references that exist, do not make up or
  hallucinate references you are not fully sure of. Only edit the `.bib` file if
  asked by the user, otherwise, let the user edit it to ensure correctness.
  References correctness is very important for the integrity of the manuscript.

* The user may write `INSTRUCTIONS.md` files; make sure the manuscript follows
  and implement these instructions. These files may contain, for example, the
  structure and content that the user wants to see in the manuscript.

* If there are discussions with the user that change the content of the
  manuscript, make sure to keep the `INSTRUCTIONS.md` up to date - the
  `INSTRUCTIONS.md` and actual `*.tex` in the same folder should not drift
  away from each other.

* The whole project is under git version control; when asked to do a task,
  check `git diff` to see if any guidelines or instructions or other have been
  updated, and make sure to follow these.

* The manuscript should be compiled with pdflatex

* When compiling or running any command to build the document, make sure there
  are no errors in the logs / check the logs, if there are some iterate on the
  source until things compile.

* Make sure the captions match the figures.

* Make sure what is written is correct and real science, not hallucination.
  Correctness is important. If in doubt, especially about the science, ask the
  human expert.

* You can and should push back / discuss points where you think the human (me
  writing this) may be wrong, but if after discussing I say I am sure, you
  should do as I say - I am an expert in the field. Keep a list of discussion
  points into the `TRUTHS.md` file at the root of the repo for when you were
  unsure but we had a discussion and I told you what to do.

* If anything is unclear, either instructions, structure, or so, start
  discussing with the human (me). Similar if you see possible improvements.

* Keep notes in the `AI_REFLECTIONS.md` file if you make important reflections
  about the manuscript or the science that you want to remember, back of the
  envelope calculations, etc.

* Be straight and to the point, dont be overly verbose, dont explain simple
  concepts etc: the human (me) is an expert with lots of experience on what is
  being written about.

* If you see possible improvements to anything, both the manuscript but also
  the "meta" stuff (like the `.md` files that you get from me), let me know /
  start a discussion.

* I may put related papers in the `related_docs` folder; these may be either
  previous papers from me you should draw inspiration from, or related papers
  that should be taken into account when writing the new paper, or whatever I say
  in the detailed `INSTRUCTIONS.md`.

* After any change, before going back to the user, make sure to ensure
  consistency of the project.


## General workflow aspects

* Keep logs of tasks done, discussions with the user, etc, in AI_LOGS.md .
  Start with the timestamp (YYYY-MM-DDTHH:MM:SSZ), then have a short entry.

* If you think anything should be improved, also the "meta" aspect of how this
  is set up, let the user know.

* The local machine on which you run may be a headless / non interactive
  virtual machine (VM); take this into account.

* The local machine should have the necessary pdflatex etc commands installed;
  if not, notify the user.

## Session startup

When starting a new session, follow these steps in order:

1. Read this file (`AGENTS.md`) fully.
2. Read all `.md` files in the project root (`README.md`, `DOCUMENTATION.md`,
   `AI_LOGS.md`, `AI_REFLECTIONS.md`, `TRUTHS.md`); skip those that do not exist.
3. Inspect the folder structure and read the existing code and `report/main.tex`.
4. Check that `pdflatex` and, if the Python tooling below is used, the mamba
   environment exist; install only after asking the user.
5. Build the report (`./build.sh`) to confirm the baseline state.
6. Present a plan and wait for approval before changing files (see "Guidelines
   for the writing process").

## Session wrap-up

At the end of each session that fulfils a plan (not just a small edit), update
`DOCUMENTATION.md` with newly reached conclusions, resolved and new open
questions, and concrete next steps for the following session, in addition to
the "Documentation clean-up" below.

## Documentation clean-up

Do this at the end of every session that fulfils a plan, and whenever the user
asks for a documentation clean-up. Present a plan first (see "Guidelines for
the writing process"). Do all of the following:

1. `README.md`: update Status and "Not done yet", and make Layout list every
   script, data file and folder.
2. `DOCUMENTATION.md`: update Sources, Budget method, Key assumptions and
   Status/TODO; keep one section per script in run order; remove stale or
   duplicated text.
3. `report/main.tex`: update it to match `DOCUMENTATION.md`, `TRUTHS.md` and
   `data/assumptions.csv`; rebuild with `./build.sh`; check the log for `!`
   errors and for new Overfull warnings.
4. `environment.yml` (if it exists): pin every package added during the
   session, using the versions installed in the env.
5. `.gitignore`: add newly generated logs, work directories and outputs.
6. `AI_LOGS.md`: add a short entry in the format below.
7. `AI_REFLECTIONS.md`: update only if there was new complex reasoning.
8. `TRUTHS.md`: add points the user decided after discussion.
9. Check that no file contradicts another and that none contains secrets or
   confidential partner data.
10. Tell the user what was changed and what could not be verified.

Format of `AI_LOGS.md` entries (short and scannable, not a transcript):

```
## YYYY-MM-DDTHH:MM:SSZ — <one-line topic>
**Prompt:** <one or two sentence summary of what the user asked>
**Actions:** <bullet list of what was done>
**Outcome:** <one sentence on result or decision reached>
```

## Python practices (scripts/)

* Simple, idiomatic Python, PEP-8, max line length 99 for code and 70 for
  comments and docstrings; type annotations and Google-style docstrings.
* Magic constants as ALL_CAPS at the top of the file; `if __name__ == "__main__":`
  guard; explicit exit codes for scripts.
* Check with `codespell`, `ruff`, `mypy` and `py_compile`; use pytest for
  serious functions only, not for small scripts.
* Standard library only unless the user agrees to more packages.
* Code defensively: assert assumptions about input data (e.g. `data/assumptions.csv`).
* Use mamba with conda-forge only (no paid Anaconda channels), pinned versions in
  `environment.yml`, `-y` for non-interactive installs; do not install mamba
  without asking.

## Data and scope

* If data is missing or something is ambiguous, ask the user; do not create
  workarounds or fetch data from the web.
* Avoid long, expensive processes; suggest a simpler alternative first.
* Never change anything outside this folder (sibling repositories, the user's
  other projects); track issues in the relevant `.md` files instead.

