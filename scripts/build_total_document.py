from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUARTO_DIR = ROOT / "quarto"
EXAM_DIR = ROOT / "exam"
OUTPUT = QUARTO_DIR / "CS114_Total_Document.qmd"

FULL_NOTES = QUARTO_DIR / "CS114_Full_Notes.qmd"

EXAM_FILES = [
    EXAM_DIR / "sample_exam" / "CS114 - Exam_Recovered.md",
    EXAM_DIR / "sample_exam" / "CS114 - Exam_Answer.md",
    EXAM_DIR / "exam_01" / "CS114 - Exam_01.md",
    EXAM_DIR / "exam_01" / "CS114 - Exam_01_Answer.md",
    EXAM_DIR / "exam_02" / "CS114 - Exam_02.md",
    EXAM_DIR / "exam_02" / "CS114 - Exam_02_Answer.md",
    EXAM_DIR / "exam_03" / "CS114 - Exam_03.md",
    EXAM_DIR / "exam_03" / "CS114 - Exam_03_Answer.md",
    EXAM_DIR / "exam_04" / "CS114 - Exam_04.md",
    EXAM_DIR / "exam_04" / "CS114 - Exam_04_Answer.md",
    EXAM_DIR / "exam_05" / "CS114 - Exam_05.md",
    EXAM_DIR / "exam_05" / "CS114 - Exam_05_Answer.md",
]


def strip_yaml(text: str) -> str:
    if not text.startswith("---\n"):
        return text.lstrip()

    end = text.find("\n---", 4)
    if end == -1:
        return text.lstrip()

    return text[end + len("\n---") :].lstrip()


def demote_headings(text: str, levels: int = 1) -> str:
    demoted: list[str] = []
    prefix = "#" * levels
    for line in text.splitlines():
        if line.startswith("#"):
            demoted.append(f"{prefix}{line}")
        else:
            demoted.append(line)
    return "\n".join(demoted).strip()


def main() -> None:
    missing = [path for path in [FULL_NOTES, *EXAM_FILES] if not path.exists()]
    if missing:
        joined = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required file(s):\n{joined}")

    notes_body = strip_yaml(FULL_NOTES.read_text(encoding="utf-8")).strip()

    exam_sections: list[str] = []
    for index, path in enumerate(EXAM_FILES, start=1):
        body = strip_yaml(path.read_text(encoding="utf-8"))
        body = demote_headings(body, levels=1)
        rel_path = path.relative_to(ROOT).as_posix()
        title = path.stem
        exam_sections.append(
            "\\clearpage\n\n"
            f"# Exam Appendix {index}: {title}\n\n"
            f"*Source: `{rel_path}`*\n\n"
            f"{body}\n"
        )

    exams_body = "\n\n".join(exam_sections)

    combined = f"""---
title: "CS114: Complete Notes and Exam Pack"
author: "Quoc Kien"
jupyter: science_env
number-sections: false
toc: true
toc-depth: 3
toc-title: "Detailed table of contents"
format:
  html:
    theme: cosmic
    toc: true
    toc-depth: 3
    number-sections: false
  pdf:
    documentclass: scrartcl
    toc: true
    toc-depth: 3
    number-sections: false
    include-in-header:
      text: |
        \\usepackage{{amsmath}}
        \\usepackage{{mathtools}}
        \\usepackage{{fvextra}}
        \\DefineVerbatimEnvironment{{Highlighting}}{{Verbatim}}{{breaklines,breakanywhere,commandchars=\\\\\\{{\\}}}}
        \\allowdisplaybreaks
        \\setlength{{\\emergencystretch}}{{3em}}
        \\AtBeginDocument{{\\hypersetup{{bookmarksopen=true,bookmarksnumbered=true,bookmarksdepth=3}}}}
---

# How to Use This Complete Pack

This document joins the full CS114 notes with every exam and answer key in the `exam/` directory.

- Part 1 contains the complete notebook-based knowledgebase.
- Part 2 contains the recovered sample exam, five practice exams, and all answer keys.
- Use the PDF table of contents or bookmarks for fast traversal.

{notes_body}

\\clearpage

# Part 10: Exam Papers and Answer Keys

The following appendix joins every exam Markdown file currently present in `exam/`.

{exams_body}

\\clearpage

# Cheers

Cheers, and good luck on the exam.
"""

    OUTPUT.write_text(combined, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Included full notes plus {len(EXAM_FILES)} exam file(s).")


if __name__ == "__main__":
    main()
