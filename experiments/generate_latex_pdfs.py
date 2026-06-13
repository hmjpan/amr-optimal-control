"""Generate publication-style PDFs with Pandoc + XeLaTeX.

The manuscript Markdown uses [INSERT Fig*.png] placeholders for Word generation.
This script converts those placeholders to real Markdown image references in a
build directory, then calls pandoc with xelatex so the PDFs contain figures and
native tables.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "real_data" / "figures"
BUILD = ROOT / "outputs" / "submission_build"
BUILD.mkdir(parents=True, exist_ok=True)

MAIN_MD = ROOT / "MANUSCRIPT.md"
SUPP_MD = ROOT / "SUPPLEMENTARY_APPENDIX.md"

MAIN_PDF = ROOT / "MANUSCRIPT.pdf"
SUPP_PDF = ROOT / "SUPPLEMENTARY_APPENDIX.pdf"

MAIN_FIG_CAPTIONS = {
    "Fig1_PhaseDiagram.png": "Figure 1. R0_r bifurcation and optimal carbapenem-use policy tiers.",
    "Fig2_BootstrapPosterior.png": "Figure 2. Italy bootstrap posterior distribution for the optimal antibiotic-use policy.",
}

SUPP_FIG_CAPTIONS = {
    "SupFig1_AuxiliaryPhaseDiagram.png": "Supplementary Figure S1. Auxiliary phase diagram and cost-saving relationship.",
    "SupFig2_CountryComparison.png": "Supplementary Figure S2. Terminal resistance under optimal policy versus status quo by country.",
    "SupFig3_ResistanceTimeSeries.png": "Supplementary Figure S3. Resistance trajectories across countries and pathogen-antibiotic pairs.",
    "SupFig4_R0Distribution.png": "Supplementary Figure S4. Resistant reproduction number distribution across analysed countries.",
}


def normalise_headings(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if set(s) <= {"="} and len(s) >= 5:
            continue
        if s in {"ABSTRACT", "BACKGROUND", "METHODS", "RESULTS", "DISCUSSION", "REFERENCES"}:
            out.append(f"## {s.title() if s != 'ABSTRACT' else 'Abstract'}")
        elif s in {"Background", "Methods", "Results", "Conclusion"} and i < 25:
            out.append(f"### {s}")
        elif s.startswith("APPENDIX ") or s.startswith("SUPPLEMENTARY REFERENCES"):
            out.append(f"## {s.title()}")
        elif re.match(r"^S\d+(\.\d+)?\s+", s):
            out.append(f"### {s}")
        else:
            out.append(line)
    return "\n".join(out)


def convert_insert_markers(text: str, captions: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        fig_name = match.group(1)
        fig_path = FIG_DIR / fig_name
        if not fig_path.exists():
            return f"\n**[Missing figure: {fig_name}]**\n"
        rel = fig_path.relative_to(ROOT).as_posix()
        caption = captions.get(fig_name, fig_name)
        return f"\n![{caption}]({rel}){{ width=90% }}\n"

    return re.sub(r"\[INSERT\s+(\S+\.png).*?\]", repl, text)


def append_supp_figures(text: str) -> str:
    parts = [text, "\n\n## Supplementary Figures\n"]
    for fig_name, caption in SUPP_FIG_CAPTIONS.items():
        fig_path = FIG_DIR / fig_name
        if fig_path.exists():
            rel = fig_path.relative_to(ROOT).as_posix()
            parts.append(f"\n![{caption}]({rel}){{ width=90% }}\n")
    return "\n".join(parts)


def prepare_markdown(src: Path, dst: Path, title: str, captions: dict[str, str], is_supp: bool = False) -> None:
    text = src.read_text(encoding="utf-8")
    text = normalise_headings(text)
    text = convert_insert_markers(text, captions)
    if is_supp:
        text = append_supp_figures(text)
    yaml = (
        "---\n"
        f"title: \"{title}\"\n"
        "fontsize: 10pt\n"
        "geometry: margin=1in\n"
        "mainfont: Times New Roman\n"
        "---\n\n"
    )
    dst.write_text(yaml + text, encoding="utf-8")


def run_pandoc(md: Path, pdf: Path) -> None:
    cmd = [
        "pandoc",
        str(md),
        "-o",
        str(pdf),
        "--pdf-engine=xelatex",
        "--resource-path",
        str(ROOT),
        "--standalone",
        "--toc=false",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    main_build = BUILD / "MANUSCRIPT_pandoc.md"
    supp_build = BUILD / "SUPPLEMENTARY_APPENDIX_pandoc.md"
    prepare_markdown(
        MAIN_MD,
        main_build,
        "R0_r as an Early Warning Indicator for Carbapenem-Resistant Klebsiella pneumoniae",
        MAIN_FIG_CAPTIONS,
        is_supp=False,
    )
    prepare_markdown(
        SUPP_MD,
        supp_build,
        "Supplementary Appendix",
        SUPP_FIG_CAPTIONS,
        is_supp=True,
    )
    run_pandoc(main_build, MAIN_PDF)
    run_pandoc(supp_build, SUPP_PDF)
    for path in [MAIN_PDF, SUPP_PDF]:
        print(f"Generated {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
