"""Generate LaTeX source and PDFs for manuscript and supplementary appendix.

Requires a TeX distribution such as MiKTeX. Uses xelatex when available.
All paths are resolved relative to the repository root.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "latex"
FIG_DIR = ROOT / "outputs" / "real_data" / "figures"

MANUSCRIPT_MD = ROOT / "MANUSCRIPT.md"
SUPPLEMENT_MD = ROOT / "SUPPLEMENTARY_APPENDIX.md"

MAIN_TEX = BUILD / "MANUSCRIPT.tex"
SUPP_TEX = BUILD / "SUPPLEMENTARY_APPENDIX.tex"

MAIN_PDF = ROOT / "MANUSCRIPT.pdf"
SUPP_PDF = ROOT / "SUPPLEMENTARY_APPENDIX.pdf"

MAIN_TITLE = r"$R_{0r}$ as an Early Warning Indicator for Carbapenem-Resistant \textit{Klebsiella pneumoniae}: Deriving Optimal Antibiotic Policies from European Surveillance Data (2005--2024)"
SUPP_TITLE = r"Supplementary Appendix: Optimal Antibiotic Use Policies for Carbapenem-Resistant \textit{Klebsiella pneumoniae}"

MAIN_FIG_CAPTIONS = {
    "Fig1_PhaseDiagram.png": r"$R_{0r}$ bifurcation and optimal carbapenem-use policy tiers.",
    "Fig2_BootstrapPosterior.png": r"Italy bootstrap posterior distribution for the optimal antibiotic-use policy.",
    "Fig3_PredictiveValidation.png": r"Time-slice predictive validation for Italy, trained on 2006--2019, excluding 2020--2022, and tested on 2023--2024.",
}

SUPP_FIG_CAPTIONS = {
    "SupFig1_AuxiliaryPhaseDiagram.png": r"Auxiliary phase diagram and cost-saving relationship.",
    "SupFig2_CountryComparison.png": r"Terminal resistance under optimal policy versus status quo by country.",
    "SupFig3_ResistanceTimeSeries.png": r"Resistance trajectories across countries and pathogen-antibiotic pairs.",
    "SupFig4_R0Distribution.png": r"Resistant reproduction number distribution across analysed countries.",
}

SECTION_TITLES = {"ABSTRACT", "BACKGROUND", "METHODS", "RESULTS", "DISCUSSION", "REFERENCES"}
SUBSECTION_TITLES = {
    "Data source", "Model framework", "Parameter estimation", "Policy comparison and validation",
    "Predictive validation", "Clinical illustration",
}


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = "".join(replacements.get(ch, ch) for ch in text)
    out = out.replace("R0\_r", r"$R_{0r}$")
    out = out.replace("K. pneumoniae", r"\textit{K. pneumoniae}")
    out = out.replace("Klebsiella pneumoniae", r"\textit{Klebsiella pneumoniae}")
    out = out.replace("E. coli", r"\textit{E. coli}")
    out = out.replace("S. aureus", r"\textit{S. aureus}")
    out = out.replace("--", r"--")
    return out


def is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"=", "-"} and len(stripped) >= 5


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 and not stripped.startswith("[INSERT")


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def collect_table(lines: list[str], i: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    while i < len(lines) and is_table_line(lines[i]):
        if not is_table_separator(lines[i]):
            rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
        i += 1
    return rows, i


def latex_table(rows: list[list[str]], caption: str | None = None) -> str:
    if not rows:
        return ""
    ncol = max(len(row) for row in rows)
    align = "p{0.18\\linewidth}" + "".join(["p{0.13\\linewidth}" for _ in range(ncol - 1)])
    body = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{" + align + "}", r"\toprule"]
    for idx, row in enumerate(rows):
        padded = row + [""] * (ncol - len(row))
        body.append(" & ".join(escape_latex(cell) for cell in padded) + r" \\")
        if idx == 0:
            body.append(r"\midrule")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if caption:
        body.append(r"\caption{" + escape_latex(caption) + "}")
    body.append(r"\end{table}")
    return "\n".join(body) + "\n"


def figure_block(filename: str, captions: dict[str, str], prefix: str = "Figure") -> str:
    caption = captions.get(filename, filename)
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92\\linewidth]{{{(FIG_DIR / filename).as_posix()}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}\n"
    )


def preamble(title: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=2.3cm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\usepackage{{setspace}}
\usepackage{{titlesec}}
\hypersetup{{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}}
\captionsetup{{font=small,labelfont=bf}}
\setstretch{{1.08}}
\titleformat{{\section}}{{\large\bfseries}}{{}}{{0pt}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries}}{{}}{{0pt}}{{}}
\title{{{title}}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle
"""


def convert_markdown(md_path: Path, title: str, fig_captions: dict[str, str], is_supplement: bool = False) -> str:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = [preamble(title)]
    i = 0
    table_count = 0
    inserted = set()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or is_separator(line):
            i += 1
            continue

        insert = re.match(r"\[INSERT\s+(\S+\.png).*?\]", line)
        if insert:
            filename = insert.group(1)
            if (FIG_DIR / filename).exists():
                out.append(figure_block(filename, fig_captions))
                inserted.add(filename)
            i += 1
            continue

        if is_table_line(line):
            rows, i = collect_table(lines, i)
            table_count += 1
            caption = "Cross-country CRKP results and policy tiers" if table_count == 1 and not is_supplement else None
            out.append(latex_table(rows, caption=caption))
            continue

        if line in SECTION_TITLES:
            out.append(r"\section*{" + escape_latex(line.title()) + "}")
        elif line in SUBSECTION_TITLES:
            out.append(r"\subsection*{" + escape_latex(line) + "}")
        elif line.startswith("APPENDIX ") or line.startswith("SUPPLEMENTARY REFERENCES"):
            out.append(r"\section*{" + escape_latex(line.title()) + "}")
        elif re.match(r"^S\d+(\.\d+)?\s+", line):
            out.append(r"\subsection*{" + escape_latex(line) + "}")
        elif line.startswith("[") and "]" in line and not line.startswith("[INSERT"):
            out.append(escape_latex(line) + r"\\")
        else:
            out.append(escape_latex(line) + "\n")
        i += 1

    if is_supplement:
        out.append(r"\clearpage")
        out.append(r"\section*{Supplementary Figures}")
        for filename in fig_captions:
            if (FIG_DIR / filename).exists():
                out.append(figure_block(filename, fig_captions, prefix="Supplementary Figure"))

    out.append(r"\end{document}")
    return "\n".join(out)


def compile_tex(tex_path: Path, output_pdf: Path) -> None:
    compiler = shutil.which("xelatex") or shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("No LaTeX compiler found. Please ensure MiKTeX xelatex is on PATH.")
    for _ in range(2):
        subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    generated = tex_path.with_suffix(".pdf")
    output_pdf.write_bytes(generated.read_bytes())


def verify_figures(captions: dict[str, str]) -> None:
    missing = [name for name in captions if not (FIG_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing figures: " + ", ".join(missing))


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    verify_figures(MAIN_FIG_CAPTIONS)
    verify_figures(SUPP_FIG_CAPTIONS)

    MAIN_TEX.write_text(convert_markdown(MANUSCRIPT_MD, MAIN_TITLE, MAIN_FIG_CAPTIONS), encoding="utf-8")
    SUPP_TEX.write_text(convert_markdown(SUPPLEMENT_MD, SUPP_TITLE, SUPP_FIG_CAPTIONS, is_supplement=True), encoding="utf-8")

    compile_tex(MAIN_TEX, MAIN_PDF)
    compile_tex(SUPP_TEX, SUPP_PDF)

    print(f"Generated {MAIN_TEX.relative_to(ROOT)}")
    print(f"Generated {SUPP_TEX.relative_to(ROOT)}")
    print(f"Generated {MAIN_PDF.relative_to(ROOT)} ({MAIN_PDF.stat().st_size // 1024} KB)")
    print(f"Generated {SUPP_PDF.relative_to(ROOT)} ({SUPP_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
