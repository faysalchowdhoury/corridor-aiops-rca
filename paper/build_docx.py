"""Regenerate paper/main.docx from paper/main.tex with pandoc.

The existing paper/main.docx is used as the pandoc reference document, so the updated
file keeps the reviewed document's styles. Generated \\input files (tables and number
macros) are inlined, and PDF figures are replaced with their 350 DPI PNG versions.

Usage: python paper/build_docx.py [--pandoc PATH]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DEFAULT_PANDOC = Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"


def _inline_inputs(text: str, base: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        target = (base / match.group(1)).resolve()
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        return _inline_inputs(target.read_text(encoding="utf-8"), target.parent)

    return re.sub(r"\\input\{([^}]+)\}", replace, text)


def _closing_brace(text: str, opening: int) -> int:
    """Index of the brace that closes the group opened at ``text[opening]``."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{" and text[index - 1] != "\\":
            depth += 1
        elif text[index] == "}" and text[index - 1] != "\\":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces in caption")


def _number_floats(tex: str) -> str:
    """Number tables and figures as elsarticle does, and resolve refs."""
    numbers: dict[str, str] = {}
    counters = {"table": 0, "figure": 0}

    def number(match: re.Match[str]) -> str:
        environment, body = match.group(1), match.group(2)
        kind = "table" if "table" in environment else "figure"
        counters[kind] += 1
        count = counters[kind]
        tag = f"Table {count}." if kind == "table" else f"Fig. {count}."
        label = re.search(r"\\label\{([^}]+)\}", body)
        if label:
            numbers[label.group(1)] = str(count)
        if kind == "table":
            # pandoc drops captions of tabulars nested in size groups, so the caption
            # becomes an explicit paragraph above the table.
            start = body.index("\\caption{") + len("\\caption")
            end = _closing_brace(body, start)
            caption = body[start + 1 : end]
            body = body[: body.index("\\caption{")] + body[end + 1 :]
            body = re.sub(r"\{\\(?:footnotesize|scriptsize|small)\s*", "{", body)
            return f"\n\n\\noindent\\textbf{{{tag}}} {caption}\n\n{body}\n\n"
        body = body.replace("\\caption{", f"\\caption{{{tag} ", 1)
        return f"\\begin{{{environment}}}{body}\\end{{{environment}}}"

    tex = re.sub(
        r"\\begin\{(table\*?|sidewaystable|figure\*?)\}(.*?)\\end\{\1\}", number, tex, flags=re.S
    )
    return re.sub(r"\\ref\{([^}]+)\}", lambda match: numbers.get(match.group(1), "?"), tex)


def _braced(text: str, start: int) -> tuple[str, int]:
    """Content of the brace group opening at ``text[start]`` and the index after it."""
    end = _closing_brace(text, start)
    return text[start + 1 : end], end + 1


def _frontmatter(tex: str) -> str:
    """Rewrite the elsarticle front matter into commands pandoc understands."""
    tex = re.sub(r"\\corref\{[^}]*\}", "*", tex)
    affiliations: dict[str, str] = {}
    while (match := re.search(r"\\affiliation\[([^\]]+)\]", tex)) is not None:
        content, after = _braced(tex, match.end())
        fields = dict(re.findall(r"(\w+)=\{((?:[^{}]|\{[^{}]*\})*)\}", content))
        parts = [fields.get(key, "") for key in ("organization", "city", "state", "country")]
        affiliations[match.group(1)] = ", ".join(part for part in parts if part)
        tex = tex[: match.start()] + tex[after:].lstrip("\n")
    authors = []
    while (match := re.search(r"\\author\[([^\]]+)\]", tex)) is not None:
        name, after = _braced(tex, match.end())
        authors.append(f"{name} ({affiliations.get(match.group(1), '')})")
        tex = tex[: match.start()] + tex[after:].lstrip("\n")
    emails = []
    for command in ("ead", "cortext", "journal"):
        while (match := re.search(rf"\\{command}(\[[^\]]*\])?(?=\{{)", tex)) is not None:
            content, after = _braced(tex, match.end())
            if command == "ead":
                emails.append(content)
            tex = tex[: match.start()] + tex[after:].lstrip("\n")
    if authors:
        note = f" \\and *Corresponding author: {', '.join(emails)}" if emails else ""
        names = " \\and ".join(authors)
        title_end = tex.index("\\begin{abstract}")
        tex = tex[:title_end] + f"\\author{{{names}{note}}}\n\n" + tex[title_end:]
    tex = re.sub(r"\\(begin|end)\{frontmatter\}\n", "", tex)
    tex = re.sub(
        r"\\begin\{highlights\}(.*?)\\end\{highlights\}",
        r"\\noindent\\textbf{Highlights}\n\\begin{itemize}\1\\end{itemize}",
        tex,
        flags=re.S,
    )
    return re.sub(
        r"\\begin\{keyword\}\s*(.*?)\s*\\end\{keyword\}",
        lambda match: "\\noindent\\textbf{Keywords:} " + match.group(1).replace(" \\sep ", "; "),
        tex,
        flags=re.S,
    )


def _prepare(tex: str) -> str:
    tex = _frontmatter(tex)
    tex = re.sub(r"\\(begin\{adjustbox\}\{[^}]*\}|end\{adjustbox\})\n?", "", tex)
    tex = tex.replace(">{\\raggedright\\arraybackslash}", "")
    tex = re.sub(
        r"\*\{(\d+)\}\{(p\{[^}]*\})\}",
        lambda match: match.group(2) * int(match.group(1)),
        tex,
    )
    tex = re.sub(r"\\makeatletter.*?\\makeatother\n", "", tex, flags=re.S)
    tex = tex.replace("\\bstctlcite{NoRepeatedAuthorDash}\n", "")
    tex = re.sub(r"(figures/[A-Za-z0-9_]+)\.pdf", r"\1.png", tex)
    tex = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{(.*?)\}\n", r"\1\n", tex, flags=re.S)
    tex = re.sub(r"\\cmidrule(\([a-z]+\))?\{[^}]*\}", "", tex)
    return _number_floats(tex)


def build(pandoc: Path, output: Path = PAPER / "main.docx") -> Path:
    tex = _prepare(_inline_inputs((PAPER / "main.tex").read_text(encoding="utf-8"), PAPER))
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        source = work / "main_flat.tex"
        source.write_text(tex, encoding="utf-8")
        command = [
            str(pandoc),
            str(source),
            "--from=latex",
            "--to=docx",
            "--citeproc",
            # Paths are relative to PAPER (the working directory below): pandoc copies the
            # --csl value into the document properties, so an absolute path would leak.
            "--bibliography=references.bib",
            "--resource-path=.",
            *(["--csl=ieee.csl"] if (PAPER / "ieee.csl").exists() else []),
            "--output",
            str(work / "main.docx"),
        ]
        if output.exists():
            reference = work / "reference.docx"
            shutil.copyfile(output, reference)
            command.insert(-2, f"--reference-doc={reference}")
        subprocess.run(command, check=True, cwd=PAPER)
        shutil.copyfile(work / "main.docx", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pandoc", type=Path, default=DEFAULT_PANDOC)
    args = parser.parse_args()
    pandoc = args.pandoc if args.pandoc.exists() else Path(shutil.which("pandoc") or "pandoc")
    print(build(pandoc))


if __name__ == "__main__":
    main()
