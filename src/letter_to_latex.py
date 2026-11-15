"""Convert the rendered response letter (Markdown) to LaTeX for a PDF.

Handles the subset of Markdown the letter uses: headings (#, ##), paragraphs,
**bold**, *italic*, `code`, nested "- " lists, pipe tables and "---" rules.
Run src.render_response first.

Usage: python -m src.letter_to_latex   (writes paper/revision/response_to_reviewers.tex)
"""

import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parent.parent / "paper" / "revision"

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,textcomp}
\usepackage[margin=2.5cm]{geometry}
\usepackage{enumitem,booktabs,parskip}
\usepackage[hidelinks]{hyperref}
\setlist{nosep,leftmargin=1.5em}
\sloppy\emergencystretch=3em
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{230A}{\ensuremath{\lfloor}}
\DeclareUnicodeCharacter{230B}{\ensuremath{\rfloor}}
\DeclareUnicodeCharacter{2208}{\ensuremath{\in}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{2079}{\textsuperscript{9}}
\DeclareUnicodeCharacter{2070}{\textsuperscript{0}}
\DeclareUnicodeCharacter{00B9}{\textsuperscript{1}}
\DeclareUnicodeCharacter{00B2}{\textsuperscript{2}}
\DeclareUnicodeCharacter{00B3}{\textsuperscript{3}}
\DeclareUnicodeCharacter{2074}{\textsuperscript{4}}
\DeclareUnicodeCharacter{2075}{\textsuperscript{5}}
\DeclareUnicodeCharacter{2078}{\textsuperscript{8}}
\DeclareUnicodeCharacter{207B}{\textsuperscript{\ensuremath{-}}}
\DeclareUnicodeCharacter{00B7}{\textperiodcentered}
\begin{document}
"""

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(text):
    return "".join(SPECIAL.get(c, c) for c in text)


def inline(text):
    # split out code spans first so their contents are only escaped
    parts = re.split(r"(`[^`]*`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{" + esc(p[1:-1]) + "}")
            continue
        p = esc(p)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", p)
        out.append(p)
    return "".join(out)


def convert(md):
    lines = md.splitlines()
    out, para, depth = [], [], []   # depth: stack of open itemize indents
    i = 0

    def flush_para():
        if para:
            out.append(inline(" ".join(para)) + "\n")
            para.clear()

    def close_lists(to=-1):
        while depth and depth[-1] > to:
            out.append(r"\end{itemize}")
            depth.pop()

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^( *)- (.*)$", line)
        if m:
            flush_para()
            ind = len(m.group(1))
            if not depth or ind > depth[-1]:
                out.append(r"\begin{itemize}")
                depth.append(ind)
            else:
                close_lists(ind)
            out.append(r"\item " + inline(m.group(2)))
            i += 1
            continue
        if line.strip() == "":
            flush_para()
            close_lists()
            i += 1
            continue
        close_lists()
        if line.startswith("# "):
            flush_para()
            out.append(r"{\Large\bfseries " + inline(line[2:]) + r"\par}\medskip")
        elif line.startswith("## "):
            flush_para()
            out.append(r"\section*{" + inline(line[3:]) + "}")
        elif line.strip() == "---":
            flush_para()
            out.append(r"\medskip\noindent\rule{\linewidth}{0.4pt}\par\medskip")
        elif line.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            ncol = len(rows[0])
            out.append(r"\begin{center}\small\begin{tabular}{ll" + "r" * (ncol - 2) + r"}\toprule")
            out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in rows[0]) + r" \\\midrule")
            for r in rows[1:]:
                out.append(" & ".join(inline(c) for c in r) + r" \\")
            out.append(r"\bottomrule\end{tabular}\end{center}")
            continue
        elif line.startswith("**") and line.rstrip().endswith("  ") is False and para == [] \
                and re.match(r"^\*\*[^*]+:\*\* ", line):
            # header lines such as "**Journal:** ..." each on their own line
            out.append(inline(line) + r"\\")
        else:
            para.append(line.strip())
        i += 1
    flush_para()
    close_lists()
    return PREAMBLE + "\n".join(out) + "\n\\end{document}\n"


def main():
    md = (REV / "response_to_reviewers.md").read_text()
    if "<<" in md:
        print("letter has unresolved tokens; run src.render_response first")
        return 1
    (REV / "response_to_reviewers.tex").write_text(convert(md))
    print("wrote response_to_reviewers.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
