"""Convert the research manuscript to engine-compatible LaTeX source.

The English Markdown manuscript remains the content source. This converter uses
only the Python standard library and writes one UTF-8 .tex file; compilation is
a separate step. Figures remain linked to the existing project diagrams.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "AquaPulse-Research-Report.en.md"
DESTINATION = ROOT / "AquaPulse-Research-Report.tex"

PREAMBLE = r"""% Generated from AquaPulse-Research-Report.en.md. Edit the Markdown and regenerate.
% Compile from docs/submission with pdfLaTeX, XeLaTeX, or LuaLaTeX.
\documentclass[a4paper,11pt]{article}
\usepackage[a4paper,margin=22mm,footskip=11mm]{geometry}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{newtxtext}
\else
  \usepackage{fontspec}
  \setmainfont{texgyretermes}[
    Extension=.otf,
    UprightFont=*-regular,
    BoldFont=*-bold,
    ItalicFont=*-italic,
    BoldItalicFont=*-bolditalic
  ]
\fi
\usepackage{amsmath}
\usepackage{newtxmath}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{caption}
\usepackage{xurl}
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{titlesec}

% Plain page numbers in the footer. No running title or page-header rule.
\pagestyle{plain}
\raggedbottom
\setlength{\parindent}{1.4em}
\setlength{\parskip}{0.25em}
\setlength{\emergencystretch}{2em}
\clubpenalty=1000
\widowpenalty=1000
\displaywidowpenalty=1000
\titleformat{\section}{\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.6em}{}
\titlespacing*{\section}{0pt}{1.2em plus 0.3em minus 0.2em}{0.5em}
\titlespacing*{\subsection}{0pt}{0.9em plus 0.2em minus 0.1em}{0.4em}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false}
\captionsetup[table]{position=top,skip=6pt}
\captionsetup[figure]{skip=6pt}
\setlength{\textfloatsep}{11pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{3}
\renewcommand{\topfraction}{0.88}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.10}
\renewcommand{\floatpagefraction}{0.65}
\urlstyle{same}
\hypersetup{
  pdftitle={AquaPulse: Citizen Observations and Human Verification for Urban River Monitoring},
  pdfauthor={AquaPulse project},
  pdfsubject={Technical research report and prototype evaluation},
  pdfkeywords={urban rivers, citizen observations, YOLO11, Kimi, human verification}
}
\begin{document}
"""

SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
    "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
    "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    "≤": r"\(\leq\)", "≥": r"\(\geq\)", "×": r"\(\times\)",
}
INLINE = re.compile(
    r"https?://[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"`[^`]+`|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|\[[1-9]\d*\]"
)


def escape(text: str) -> str:
    return "".join(SPECIAL.get(character, character) for character in text)


def inline(text: str, *, citations: bool = True) -> str:
    """Escape prose while retaining URLs, emails, emphasis, and numbered cites."""
    result, cursor = [], 0
    for match in INLINE.finditer(text):
        result.append(escape(text[cursor:match.start()]))
        token = match.group()
        if token.startswith(("https://", "http://")):
            # Sentence punctuation belongs outside the hyperlink.
            url = token.rstrip(".,;")
            result.append(r"\url{" + url + "}" + escape(token[len(url):]))
        elif "@" in token and not token.startswith(("`", "*")):
            result.append(r"\href{mailto:" + token + "}{" + escape(token) + "}")
        elif token.startswith("`"):
            result.append(r"\texttt{" + escape(token[1:-1]) + "}")
        elif token.startswith("**"):
            result.append(r"\textbf{" + inline(token[2:-2], citations=citations) + "}")
        elif token.startswith("*"):
            result.append(r"\emph{" + inline(token[1:-1], citations=citations) + "}")
        elif citations:
            result.append(r"\cite{ref" + token[1:-1] + "}")
        else:
            result.append(escape(token))
        cursor = match.end()
    result.append(escape(text[cursor:]))
    return "".join(result)


@dataclass(frozen=True)
class Block:
    kind: str
    text: str = ""
    level: int = 0
    caption: str = ""
    path: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    alignment: tuple[str, ...] = ()
    number: int = 0


def split_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def parse(source: str) -> list[Block]:
    """Parse all nonblank lines, refusing unpaired captions or malformed tables."""
    lines = source.splitlines()
    blocks, pending_caption = [], ""
    consumed, index = set(), 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        consumed.add(index)
        heading = re.fullmatch(r"(#{1,3})\s+(.+)", line)
        figure = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
        caption = re.fullmatch(r"Table (\d+)\. (.+)", line)
        if heading:
            blocks.append(Block("heading", heading[2], level=len(heading[1])))
        elif figure:
            numbered = re.fullmatch(r"Figure (\d+)\. (.+)", figure[1])
            if not numbered:
                raise ValueError("Each figure must retain its numbered source caption.")
            if not (SOURCE.parent / figure[2]).is_file():
                raise FileNotFoundError(figure[2])
            blocks.append(Block("figure", caption=numbered[2], path=figure[2], number=int(numbered[1])))
        elif caption:
            if pending_caption:
                raise ValueError("Table caption has no following table.")
            pending_caption = line
        elif line.startswith("|"):
            if not pending_caption:
                raise ValueError("Table has no source caption.")
            rows, alignment = [], ()
            while index < len(lines) and lines[index].strip().startswith("|"):
                consumed.add(index)
                cells = split_cells(lines[index])
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    alignment = tuple("r" if cell.endswith(":") else "l" for cell in cells)
                else:
                    rows.append(cells)
                index += 1
            if not rows or not alignment or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Malformed Markdown table.")
            numbered = re.fullmatch(r"Table (\d+)\. (.+)", pending_caption)
            assert numbered is not None
            blocks.append(Block("table", caption=numbered[2], rows=tuple(rows),
                                alignment=alignment, number=int(numbered[1])))
            pending_caption = ""
            continue
        elif line.startswith("$$") and line.endswith("$$"):
            expression = line[2:-2].strip()
            tag = re.search(r"\s*\\qquad\s+\((\d+)\)$", expression)
            if not tag:
                raise ValueError("Equation has no source number.")
            blocks.append(Block("equation", text=expression[:tag.start()].strip(), number=int(tag[1])))
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                consumed.add(index)
                paragraph.append(lines[index].strip())
                index += 1
            text = " ".join(paragraph)
            reference = re.fullmatch(r"\[(\d+)\]\s+(.+)", text)
            blocks.append(Block("reference" if reference else "paragraph",
                                text=reference[2] if reference else text,
                                number=int(reference[1]) if reference else 0))
            continue
        index += 1
    if pending_caption:
        raise ValueError("Unpaired final table caption.")
    assert consumed == {n for n, line in enumerate(lines) if line.strip()}, "Source content omitted."
    return blocks


def table_tex(block: Block, note: str | None = None) -> str:
    # The first descriptive column absorbs spare space; original Markdown numeric
    # alignment is retained. Table 2's model column stays left aligned as text.
    specification = "@{}>{\\raggedright\\arraybackslash}X" + "".join(block.alignment[1:]) + "@{}"
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + inline(block.caption) + "}",
             r"\label{tab:" + str(block.number) + "}", r"\fontsize{9.5}{11.5}\selectfont",
             r"\setlength{\tabcolsep}{4.3pt}", r"\renewcommand{\arraystretch}{1.18}",
             r"\begin{tabularx}{\linewidth}{" + specification + "}", r"\toprule"]
    for index, row in enumerate(block.rows):
        cells = [inline(cell, citations=False) for cell in row]
        if index == 0:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        lines.append(" & ".join(cells) + r" \\")
        if index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    if note is not None:
        lines.extend([r"\par\vspace{4pt}", r"\begin{minipage}{\linewidth}",
                      r"\small\textit{" + inline(note) + "}", r"\end{minipage}"])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def convert(source: str) -> tuple[str, dict]:
    blocks = parse(source)
    assert blocks[0].kind == "heading" and blocks[0].level == 1
    figures = [block for block in blocks if block.kind == "figure"]
    tables = [block for block in blocks if block.kind == "table"]
    references = [block for block in blocks if block.kind == "reference"]
    assert [block.number for block in figures] == [1, 2, 3, 4]
    assert [block.number for block in tables] == [1, 2, 3, 4, 5]
    assert references and [block.number for block in references] == list(range(1, len(references) + 1))
    lines = [PREAMBLE, "% Source SHA-256: " + hashlib.sha256(source.encode()).hexdigest(),
             r"\begin{center}", r"{\LARGE\bfseries " + inline(blocks[0].text) + r"\par}",
             r"\vspace{0.5em}"]
    index = 1
    while blocks[index].kind == "paragraph":
        lines.append(r"{\small " + inline(blocks[index].text) + r"\par}")
        index += 1
    lines.extend([r"\end{center}", r"\vspace{0.2em}"])
    in_abstract = in_references = False
    emitted = index
    attached_notes: set[int] = set()
    for block_index, block in enumerate(blocks[index:], start=index):
        if block_index in attached_notes:
            emitted += 1
            continue
        if block.kind == "heading":
            if in_abstract:
                lines.append(r"\end{abstract}")
                in_abstract = False
            if block.text == "Abstract":
                lines.append(r"\begin{abstract}")
                in_abstract = True
            elif block.text == "References":
                # Pending floats may flow ahead of the bibliography without a forced
                # blank page. LaTeX supplies the original numbered reference labels.
                lines.extend([r"\begin{thebibliography}{" + str(len(references)) + "}", r"\small",
                              r"\setlength{\itemsep}{0.35em}"])
                in_references = True
            else:
                title = re.sub(r"^\d+(?:\.\d+)*\.\s+", "", block.text)
                command = "section" if block.level == 2 else "subsection"
                lines.append("\\" + command + "{" + inline(title) + "}")
        elif block.kind == "paragraph":
            if in_abstract and block.text.startswith("Keywords:"):
                lines.append(r"\end{abstract}")
                in_abstract = False
                lines.append(r"\noindent\textit{" + inline(block.text) + "}")
            else:
                if in_references:
                    lines.append(r"\end{thebibliography}")
                    in_references = False
                if block.text.startswith("*Different test set:"):
                    lines.append(r"{\small\noindent\textit{" + inline(block.text) + "}\\par}")
                else:
                    lines.append(inline(block.text))
        elif block.kind == "figure":
            lines.extend([r"\begin{figure}[!htbp]", r"\centering",
                          r"\includegraphics[width=0.94\linewidth,height=0.38\textheight,keepaspectratio]{" + block.path + "}",
                          r"\caption{" + inline(block.caption) + "}",
                          r"\label{fig:" + str(block.number) + "}", r"\end{figure}"])
        elif block.kind == "table":
            note = None
            if block_index + 1 < len(blocks):
                next_block = blocks[block_index + 1]
                if next_block.kind == "paragraph" and next_block.text.startswith("*Different test set:"):
                    note = next_block.text
                    attached_notes.add(block_index + 1)
            lines.append(table_tex(block, note))
        elif block.kind == "equation":
            lines.extend([r"\begin{equation}", block.text, r"\tag{" + str(block.number) + "}", r"\end{equation}"])
        elif block.kind == "reference":
            assert in_references
            lines.append(r"\bibitem{ref" + str(block.number) + "} " + inline(block.text, citations=False))
        else:
            raise ValueError(block.kind)
        lines.append("")
        emitted += 1
    if in_abstract:
        lines.append(r"\end{abstract}")
    if in_references:
        lines.append(r"\end{thebibliography}")
    lines.append(r"\end{document}")
    assert emitted == len(blocks), "A parsed block was not emitted."
    document = "\n".join(lines) + "\n"
    assert document.count(r"\begin{figure}") == len(figures)
    assert document.count(r"\begin{table}") == len(tables)
    assert document.count(r"\bibitem{") == len(references)
    cited = {int(number) for number in re.findall(r"\\cite\{ref(\d+)\}", document)}
    assert cited == {block.number for block in references}, "Every reference must be cited, and every citation must resolve."
    assert "fancyhdr" not in document and r"\raggedright" not in document.split(r"\begin{document}")[0]
    return document, {
        "figures": len(figures), "tables": len(tables), "references": len(references),
        "equations": sum(block.kind == "equation" for block in blocks),
        "source_words": len(re.findall(r"\b\S+\b", source)),
        "source_blocks": len(blocks), "emitted_blocks": emitted,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "justified_prose": True, "running_header": False,
        "figure_max_height": "0.38 text height", "engine": "pdfLaTeX / XeLaTeX / LuaLaTeX",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the existing .tex matches the manuscript.")
    arguments = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    document, audit = convert(source)
    if arguments.check:
        assert DESTINATION.read_text(encoding="utf-8") == document, "Generated LaTeX is out of date."
    else:
        DESTINATION.write_text(document, encoding="utf-8", newline="\n")
    print(json.dumps({"file": DESTINATION.name, **audit}, indent=2))


if __name__ == "__main__":
    main()
