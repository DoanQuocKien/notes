# CS114 Machine Learning Notes

This repository contains a set of Quarto notebooks for studying core machine learning topics. The notes are written as exam-friendly learning documents: each notebook combines conceptual explanation, mathematical derivation, visual intuition, executable demonstrations, and hand-written worked problems.

The current language of the notebooks is Vietnamese, with English technical terms kept where they are standard in machine learning, such as Logistic Regression, GLM, K-means, and Gradient Descent.

## What This Project Is Doing

The goal is to turn dense machine learning theory into notes that are easier to study from:

- Explain the idea before the formula.
- Break long derivations into smaller algebraic steps.
- Add visual demonstrations for abstract concepts.
- Keep Python chunks executable so plots and demonstrations are reproducible.
- Write problem-set solutions by hand, not as code output, so the PDFs work as printable exam cheat sheets.
- Export every notebook to PDF in the `output/` folder.

## Notebook Set

The full combined document is available at `output/CS114_Full_Notes.pdf`. It is generated from all 9 source notebooks by `scripts/build_full_notes.py`.

| Source notebook | Main topic | Generated PDF |
| :--- | :--- | :--- |
| `quarto/LR_OLS_Derivation.qmd` | Ordinary Least Squares derivation, residual checks, simple linear regression | `output/LR_OLS_Derivation.pdf` |
| `quarto/LR_MLE_Matrix_Derivation.qmd` | Linear regression from Gaussian MLE and matrix normal equations | `output/LR_MLE_Matrix_Derivation.pdf` |
| `quarto/Classification_Logistic_Regression.qmd` | Binary classification, sigmoid, cross-entropy, logistic regression optimization | `output/Classification_Logistic_Regression.pdf` |
| `quarto/Exponential_Family_GLMs.qmd` | Exponential family, GLMs, inverse links, Bernoulli/Poisson/Gaussian cases | `output/Exponential_Family_GLMs.pdf` |
| `quarto/GLA.qmd` | Generative learning, Bayes rule, Naive Bayes, Gaussian Discriminant Analysis | `output/GLA.pdf` |
| `quarto/Decision_Tree.qmd` | Decision trees, entropy, Gini, pruning, ensemble intuition | `output/Decision_Tree.pdf` |
| `quarto/Neural_Network.qmd` | Neural networks, layer shapes, activation functions, backpropagation | `output/Neural_Network.pdf` |
| `quarto/Optimization_Bias_Variance_Regularization.qmd` | Gradient descent, bias-variance decomposition, regularization, cross-validation | `output/Optimization_Bias_Variance_Regularization.pdf` |
| `quarto/Unsupervised_Learning_K_Means.qmd` | Unsupervised learning, K-means, centroid updates, elbow method | `output/Unsupervised_Learning_K_Means.pdf` |

## Repository Layout

```text
.
+-- quarto/               # Quarto source notebooks
+-- html/                 # Rendered HTML previews and their *_files assets
+-- output/               # Rendered PDF files
+-- exam/                 # Sample and practice exams with step-by-step answer keys
+-- scripts/              # Rebuild scripts for generated source files
+-- temp/                 # Quarto/Jupyter intermediate files and scratch artifacts
+-- README.md             # Project overview
```

The `.qmd` files in `quarto/` are the source of truth. The HTML, PDF, and temp files are generated artifacts.

## Practice Exams

In addition to the notebooks, this repository contains a set of practice exams located in the `exam/` directory. These are designed to test both mechanical calculation and deep conceptual understanding:

- **sample_exam**: The original recovered exam and its detailed answer key.
- **exam_01 to exam_05**: Five additional practice exams, each containing 6 standard questions covering the full scope of the notebooks, plus a final **advanced question** designed to challenge deeper theoretical understanding (e.g., Generalization in SGD, Curse of Dimensionality, MAP estimation for Regularization, Inductive Bias).
- Every exam is accompanied by a step-by-step mathematical answer key.

## Requirements

To render the notebooks, the local setup should have:

- Quarto
- A working Python/Jupyter environment
- The `science_env` Jupyter kernel
- Common scientific Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
- A LaTeX engine for PDF rendering, such as TinyTeX

The notebooks are configured with:

```yaml
jupyter: science_env
```

So Quarto will try to execute code chunks using the `science_env` kernel.

## Rendering

Preview a single notebook:

```powershell
quarto preview .\quarto\Classification_Logistic_Regression.qmd
```

Render one notebook to PDF:

```powershell
Push-Location .\quarto
quarto render .\Classification_Logistic_Regression.qmd --to pdf --output Classification_Logistic_Regression.pdf
Pop-Location
Move-Item -LiteralPath .\quarto\Classification_Logistic_Regression.pdf -Destination .\output\Classification_Logistic_Regression.pdf -Force
```

Important: avoid using `--output-dir output` for single-file renders in this folder. In this project, the safer pattern is to render the PDF beside the source file and then move that one PDF into `output/`.

Render one notebook to HTML and move the preview artifacts:

```powershell
Push-Location .\quarto
quarto render .\Classification_Logistic_Regression.qmd --to html --output Classification_Logistic_Regression.html
Pop-Location
Move-Item -LiteralPath .\quarto\Classification_Logistic_Regression.html -Destination .\html\Classification_Logistic_Regression.html -Force
Move-Item -LiteralPath .\quarto\Classification_Logistic_Regression_files -Destination .\html\Classification_Logistic_Regression_files -Force
```

Build the full combined source and render the big PDF:

```powershell
python .\scripts\build_full_notes.py
quarto render .\quarto\CS114_Full_Notes.qmd --to pdf --output CS114_Full_Notes.pdf
Move-Item -LiteralPath .\CS114_Full_Notes.pdf -Destination .\output\CS114_Full_Notes.pdf -Force
```

The combined PDF has a detailed table of contents and PDF bookmarks. Each source notebook starts at a new page for faster navigation.

## Study Design

Each notebook is structured to support three study modes:

- **Concept mode:** read the explanations and visual intuition.
- **Formula mode:** use the boxed equations and cheat-sheet tables.
- **Exam mode:** practice the worked problems, which show written computation steps instead of relying on code.

Python chunks are kept for demonstrations and plots, but the problem-set solutions are written out manually so the notes remain useful on paper.

## Current Status

All 9 notebooks have been rendered successfully to PDF in `output/`. The Python chunks compile under `science_env`, and the worked problem sections are written as step-by-step mathematical solutions.

The combined document `output/CS114_Full_Notes.pdf` has also been rendered successfully from all 9 notebooks.
