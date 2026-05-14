# LaTeX report — guide for beginners

This project’s report lives in the `report/` folder. You edit **plain text** `.tex` files; a tool turns them into **`main.pdf`**. You do **not** need to learn all of LaTeX—copy patterns from `chapters/example.tex` and adjust the text.

---

## 1. What you need on your computer (terminal-first)

**Step 1 — see if it is already there.** Many setups already have LaTeX from a past course or from a package manager. In a terminal (including the one integrated in your IDE), run:

```bash
pdflatex --version
latexmk -v
```

If both print version information, you are fine. Your editor’s LaTeX extensions (if any) **do not replace** this: they call `pdflatex` / `latexmk` on your machine.

**Step 2 — only if a command is missing**, install a **TeX distribution** (provides `pdflatex`, `bibtex`, fonts, and LaTeX packages). Prefer the same terminal you use for everything else:

- **macOS with [Homebrew](https://brew.sh/):**  
  `brew install --cask basictex` — small install; if a build later says a package is missing, install it with TeX’s package manager (`tlmgr`) as needed. For a full offline-style install: `brew install --cask mactex`.  
  After installing BasicTeX or MacTeX, ensure TeX is on your `PATH` (often `/Library/TeX/texbin`; open a new terminal and check `which pdflatex`).
- **Linux (Debian/Ubuntu-style):** e.g. `sudo apt install texlive-latex-extra texlive-fonts-recommended latexmk` (or a larger metapackage such as `texlive-full` if you prefer fewer missing-package errors).
- **Windows:** `winget` / Chocolatey / Scikit installer for **MiKTeX** or **TeX Live**, or use the official installer from the TeX Live / MiKTeX websites if you prefer a GUI wizard.

You do **not** have to use a browser if you already use Homebrew, `apt`, or `winget`; the links above are optional fallbacks.

---

## 2. Where everything is

**Inside `report/` (the LaTeX project):** sources and build output for the PDF.

| Path | Role |
|------|------|
| `report/main.tex` | **Main file** — loads the title page, table of contents, chapters, and bibliography. Start here when adding chapters. |
| `report/title.tex` | **Title page** — course name, your name, year, optional logo. |
| `report/chapters/*.tex` | **Chapter content** — one file per chapter is a good habit (e.g. `example.tex`). |
| `report/references.bib` | **Bibliography database** — one entry per source; you cite by **key** in the text. |
| `report/figures/` | **Images** shipped with the report (e.g. PNG, PDF). Optional: `KULeuvenLogo.png` for the title page. |
| `report/main.pdf` | **Output PDF** — always produced **here** when you build successfully. |

**Elsewhere in the repo (normal for coursework):** notebooks, Python/R scripts, data, and plots **do not** have to live under `report/`. The template already adds **`eda_plots/`** at the repository root to the search path for figures (see `\graphicspath` in `main.tex`). For another folder (e.g. `outputs/plots/`), add one more path there, e.g. `{../outputs/plots/}`, so `\includegraphics{myplot.png}` still works without copying files into `report/figures/`.

Auxiliary files (`main.aux`, `main.log`, `main.toc`, …) are written next to `main.tex` in `report/`; `report/.gitignore` ignores most of them so Git stays clean.

---

## 3. How to “compile” (build the PDF)

The PDF and auxiliary files are always written under **`report/`**. You can run the build **from the repository root** or **from `report/`** — both are fine if you use the right command.

**Recommended — from the AdvancedAnalytics repository root** (good when your “current task” is the whole project, not only LaTeX):

```bash
latexmk -pdf -cd report/main.tex
```

The **`-cd`** flag tells `latexmk` to switch to the directory of `main.tex` (`report/`) before running `pdflatex` and `bibtex`, so paths like `figures/` and `../eda_plots/` resolve correctly and **`report/main.pdf`** is updated in the right place.

**Alternative — from inside `report/`:**

```bash
cd report
latexmk -pdf main.tex
```

If you only run `pdflatex` once, the table of contents or citations can look wrong until you run it **again** after `bibtex`. Using `latexmk` avoids that.

Open **`report/main.pdf`** in any PDF viewer.

---

## 4. Mental model: commands and environments

- **Commands** start with `\` and often take arguments in `{curly braces}`: `\section{Introduction}`.
- Some commands are **pairs**: `\begin{itemize} ... \end{itemize}` (an **environment**).
- **`%`** starts a comment (rest of the line is ignored).
- **`&`** in the title page is used for “aligned table cells” in simple layouts; `\&` prints an ampersand.

The **document** starts at `\begin{document}` in `main.tex` and ends at `\end{document}`. Everything before that is **preamble** (packages and layout).

---

## 5. What to edit for a normal workflow

### Title page

Edit **`report/title.tex`**: faculty line, report title (`\textsc{...}`), subtitle line, your name, academic year, abstract line.  
If `figures/KULeuvenLogo.png` exists, it is used; otherwise the template shows plain “KU Leuven” text.

### Add or rename a chapter

1. In **`report/main.tex`**, find the block:

   ```latex
   \mainchapter{Example chapter}
   \input{chapters/example}
   ```

2. Change the **chapter title** in `\mainchapter{...}`.
3. Point `\input{...}` to your file, e.g. `\input{chapters/introduction}`.
4. Create **`report/chapters/introduction.tex`** (copy from `example.tex` and replace text).

`\mainchapter` is a custom command in this repo: it produces a **numbered** chapter title and adds it to the **table of contents**.

Inside a chapter file, use:

- `\section{...}` — large heading  
- `\subsection{...}` — smaller heading  
- Blank lines separate paragraphs.

### Figures

1. Prefer **not** copying large plots: keep them where your code writes them (e.g. `eda_plots/`) and ensure that folder is listed in `\graphicspath` in **`main.tex`** (already includes `figures/` and `../eda_plots/`). For a new directory, add another `{../your_folder/}` entry there.
2. Use the pattern in **`chapters/example.tex`**: `figure` environment, `\includegraphics[width=...]{filename}`, `\caption{...}`, `\label{fig:...}` (filename only, no path, if the folder is on `\graphicspath`).
3. Refer to it in text with `Figure~\ref{fig:...}`.

### Bullet lists

```latex
\begin{itemize}
\item First point
\item Second point
\end{itemize}
```

### Code listings

See **`chapters/example.tex`** for `lstlisting` and inline `\lstinline{...}`.

---

## 6. References and `references.bib`

1. Add an entry to **`report/references.bib`** (article, book, misc, …). Each entry has a **key**, e.g. `samuel1959some`.
2. In your chapter text, cite with `\cite{samuel1959some}`.
3. Build with **`latexmk -pdf -cd report/main.tex`** from the repo root, or **`latexmk -pdf`** from `report/` (or the manual `pdflatex` + `bibtex` + `pdflatex` ×2 sequence from `report/`).

**Important:** In `.bib` files, do not put stray `@` characters inside comment lines—BibTeX treats `@` as the start of an entry.

The bibliography style is set in `main.tex` as **`ieeetr`**. You can change `\bibliographystyle{...}` later if your course asks for another style (ask before changing).

---

## 7. If something goes wrong

- Read **`report/main.log`** near the end: LaTeX prints **Error** and **Warning** lines with line numbers.
- **“File not found”** for an image: check the file exists, the filename in `\includegraphics{...}` matches, and the folder is on `\graphicspath` in **`main.tex`**. If you run `pdflatex` **from the repo root** without changing directory, relative paths can break—use **`latexmk -pdf -cd report/main.tex`** or `cd report` before `pdflatex`.
- **Citation shows as `[?]`:** run the full build again (`latexmk` or `bibtex` + multiple `pdflatex`).
- **Unicode / special characters:** this template uses UTF-8 (`inputenc`). Prefer standard LaTeX quotes ``like this'' or use `\emph{...}`.

---

## 8. Git tip (stashing)

`git stash` only restores work when you run **`git stash pop`** or **`git stash apply`**. Untracked files are **not** stashed unless you use `git stash -u`. Commit or stash tracked files before switching branches if you care about keeping history clean.

---

For concrete copy-paste examples of sections, figures, citations, and listings, open **`report/chapters/example.tex`** and **`report/main.tex`** side by side with the PDF.
