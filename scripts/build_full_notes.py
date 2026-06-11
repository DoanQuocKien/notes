from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUARTO_DIR = ROOT / "quarto"
OUTPUT = QUARTO_DIR / "CS114_Full_Notes.qmd"

NOTEBOOKS = [
    "LR_OLS_Derivation.qmd",
    "LR_MLE_Matrix_Derivation.qmd",
    "Classification_Logistic_Regression.qmd",
    "Exponential_Family_GLMs.qmd",
    "GLA.qmd",
    "Decision_Tree.qmd",
    "Neural_Network.qmd",
    "Optimization_Bias_Variance_Regularization.qmd",
    "Unsupervised_Learning_K_Means.qmd",
]


def split_yaml(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    raw_yaml = text[4:end].strip()
    body = text[end + len("\n---") :].lstrip()
    metadata: dict[str, str] = {}

    for line in raw_yaml.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            metadata[key] = value

    return metadata, body


def main() -> None:
    missing = [name for name in NOTEBOOKS if not (QUARTO_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing notebook(s): {', '.join(missing)}")

    sections: list[str] = []
    for index, name in enumerate(NOTEBOOKS, start=1):
        path = QUARTO_DIR / name
        metadata, body = split_yaml(path.read_text(encoding="utf-8"))
        title = metadata.get("title", path.stem)
        body = body.strip()

        page_break = "\\clearpage\n\n"
        section = (
            f"{page_break}"
            f"# Phần {index}: {title}\n\n"
            f"*Nguồn: `{path.as_posix().replace(str(ROOT.as_posix()) + '/', '')}`*\n\n"
            f"{body}\n"
        )
        sections.append(section)

    joined_sections = "\n\n".join(sections)

    combined = f"""---
title: "CS114: Tài liệu Học máy Tổng hợp"
author: "Quoc Kien"
jupyter: science_env
number-sections: false
toc: true
toc-depth: 3
toc-title: "Mục lục chi tiết"
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

# Cách dùng tài liệu này

Tài liệu này ghép toàn bộ 9 notebook CS114 thành một bản PDF lớn. Mỗi notebook bắt đầu ở một trang mới và có heading cấp 1 riêng, nên có thể đi nhanh bằng mục lục PDF hoặc thanh bookmarks của trình đọc PDF.

Các phần bài tập vẫn được viết bằng lời giải tay: thay số, tính toán, kết luận. Các đoạn Python chỉ phục vụ minh họa, kiểm tra đạo hàm, hoặc tạo hình trực quan.

{joined_sections}
"""

    OUTPUT.write_text(combined, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Included {len(NOTEBOOKS)} notebook(s).")


if __name__ == "__main__":
    main()
