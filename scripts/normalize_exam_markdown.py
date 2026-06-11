from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EXAM_DIR = ROOT / "exam"

PDF_HEADER = r"""\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\setlist{nosep}
\AtBeginDocument{\hypersetup{bookmarksopen=true,bookmarksnumbered=true,bookmarksdepth=3}}
"""


def title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ")


def has_yaml(text: str) -> bool:
    return text.startswith("---\n") and "\n---" in text[4:]


def add_yaml(path: Path, text: str) -> str:
    if has_yaml(text):
        return text

    title = title_from_path(path)
    yaml = f"""---
title: "{title}"
author: "Quoc Kien"
toc: true
toc-depth: 3
format:
  pdf:
    documentclass: scrartcl
    toc: true
    toc-depth: 3
    geometry:
      - margin=0.8in
    include-in-header:
      text: |
{indent(PDF_HEADER, 8)}
---

"""
    return yaml + text.lstrip()


def indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else prefix for line in text.splitlines())


def normalize_math_segment(segment: str) -> str:
    segment = segment.replace(r"\_", "_")
    segment = segment.replace(r"\+", "+")
    segment = segment.replace(r"\-", "-")
    segment = segment.replace(r"\=", "=")
    segment = segment.replace(r"\\{", r"\{")
    segment = segment.replace(r"\\}", r"\}")
    segment = re.sub(r"\\\\([A-Za-z]+)", r"\\\1", segment)
    segment = re.sub(r"_\\mathcal\{([^}]+)\}", r"_{\\mathcal{\1}}", segment)
    segment = re.sub(r"\^\\mathcal\{([^}]+)\}", r"^{\\mathcal{\1}}", segment)
    return segment


def normalize_math(text: str) -> str:
    # Handles both inline and display math without trying to parse nested dollars.
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("$") and part.endswith("$"):
            parts[i] = normalize_math_segment(part)
    return "".join(parts)


def normalize_headings(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []

    for line in lines:
        stripped = line.strip()

        match = re.match(r"^Câu\s+(\d+)\\?\.\s*(.+)$", stripped)
        if match and not line.lstrip().startswith("#"):
            number, title = match.groups()
            out.append("")
            out.append("---")
            out.append("")
            out.append(f"## Câu {number}. {title.strip()}")
            continue

        # Pandoc sometimes escapes punctuation in plain Markdown; these are visual noise.
        line = line.replace(r"\.", ".")
        line = line.replace(r"\+", "+")
        line = line.replace(r"\-", "-")
        line = line.replace(r"\=", "=")
        out.append(line.rstrip())

    normalized = "\n".join(out)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip() + "\n"


def normalize_tables(text: str) -> str:
    # Keep Markdown tables simple for Pandoc/LaTeX.
    text = text.replace("| :---- |", "| :--- |")
    text = text.replace("| :---- | :---- |", "| :--- | :--- |")
    text = re.sub(r":----", ":---", text)
    return text


def normalize_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original.replace("\r\n", "\n")
    bookmark_setup = r"\AtBeginDocument{\hypersetup{bookmarksopen=true,bookmarksnumbered=true,bookmarksdepth=3}}"
    bookmark_option = "bookmarksopen=true,bookmarksnumbered=true,bookmarksdepth=3"
    cleaned_lines = []
    for line in text.splitlines():
        if bookmark_option in line:
            indent = line[: len(line) - len(line.lstrip())]
            cleaned_lines.append(indent + bookmark_setup)
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    if original.endswith("\n"):
        text += "\n"
    text = normalize_math(text)
    text = normalize_headings(text)
    text = normalize_tables(text)
    text = add_yaml(path, text)

    if text != original:
        path.write_text(text, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> None:
    changed = []
    for path in sorted(EXAM_DIR.rglob("*.md")):
        if normalize_file(path):
            changed.append(path.relative_to(ROOT).as_posix())

    print(f"Normalized {len(changed)} file(s).")
    for item in changed:
        print(f"- {item}")


if __name__ == "__main__":
    main()
