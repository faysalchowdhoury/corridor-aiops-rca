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


ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


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
    """Number tables (Roman) and figures (Arabic) as IEEEtran does, and resolve refs."""
    numbers: dict[str, str] = {}
    counters = {"table": 0, "figure": 0}

    def number(match: re.Match[str]) -> str:
        environment, body = match.group(1), match.group(2)
        kind = environment.rstrip("*")
        counters[kind] += 1
        count = counters[kind]
        tag = f"TABLE {ROMAN[count - 1]}." if kind == "table" else f"Fig. {count}."
        label = re.search(r"\\label\{([^}]+)\}", body)
        if label:
            numbers[label.group(1)] = ROMAN[count - 1] if kind == "table" else str(count)
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

    tex = re.sub(r"\\begin\{(table\*?|figure\*?)\}(.*?)\\end\{\1\}", number, tex, flags=re.S)
    return re.sub(r"\\ref\{([^}]+)\}", lambda match: numbers.get(match.group(1), "?"), tex)


def _prepare(tex: str) -> str:
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
