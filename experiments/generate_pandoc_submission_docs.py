"""Generate submission DOCX and PDF with Pandoc so LaTeX math becomes equations.

Outputs:
  MANUSCRIPT.docx / MANUSCRIPT.pdf
  SUPPLEMENTARY_APPENDIX.docx / SUPPLEMENTARY_APPENDIX.pdf

All paths are resolved relative to the repository root.
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

MAIN_DOCX = ROOT / "MANUSCRIPT.docx"
MAIN_PDF = ROOT / "MANUSCRIPT.pdf"
SUPP_DOCX = ROOT / "SUPPLEMENTARY_APPENDIX.docx"
SUPP_PDF = ROOT / "SUPPLEMENTARY_APPENDIX.pdf"

MAIN_FIG_CAPTIONS = {
    "Fig1_PhaseDiagram.png": "Figure 1. R0_r bifurcation and optimal carbapenem-use policy tiers.",
    "Fig2_BootstrapPosterior.png": "Figure 2. Italy bootstrap posterior distribution for the optimal antibiotic-use policy.",
}

SUPP_FIG_CAPTIONS = {
    "SupFig1_AuxiliaryPhaseDiagram.png": "Supplementary Figure S1. Auxiliary phase diagram and cost-saving relationship.",
    "SupFig2_CountryComparison.png": "Supplementary Figure S2. Terminal resistance under optimal policy versus status quo by country.",
    "SupFig3_ResistanceTimeSeries.png": "Supplementary Figure S3. Exploratory time-slice prediction check and resistance trajectories.",
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
        elif s in {"Background", "Methods", "Results", "Conclusion"} and i < 30:
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


def convert_display_math(text: str) -> str:
    r"""Convert LaTeX \[...\] display math to Pandoc-friendly $$...$$ blocks."""
    # Regex conversion is safer than line-state conversion because the source
    # may already contain native $$ blocks. Keep existing $$ blocks unchanged.
    return re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\n\1\n$$", text, flags=re.DOTALL)


def append_supp_figures(text: str) -> str:
    parts = [text, "\n\n## Supplementary Figures\n"]
    for fig_name, caption in SUPP_FIG_CAPTIONS.items():
        fig_path = FIG_DIR / fig_name
        if fig_path.exists():
            rel = fig_path.relative_to(ROOT).as_posix()
            parts.append(f"\n![{caption}]({rel}){{ width=90% }}\n")
    return "\n".join(parts)


def prepare(src: Path, dst: Path, title: str, captions: dict[str, str], is_supp: bool) -> None:
    text = src.read_text(encoding="utf-8")
    text = normalise_headings(text)
    text = convert_display_math(text)
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


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def pandoc(md: Path, out: Path) -> None:
    cmd = [
        "pandoc",
        "-f",
        "markdown+tex_math_single_backslash+tex_math_dollars",
        str(md),
        "-o",
        str(out),
        "--resource-path",
        str(ROOT),
        "--standalone",
    ]
    if out.suffix.lower() == ".pdf":
        cmd.extend(["--pdf-engine=xelatex"])
    run(cmd)


def main() -> None:
    main_build = BUILD / "MANUSCRIPT_pandoc.md"
    supp_build = BUILD / "SUPPLEMENTARY_APPENDIX_pandoc.md"
    prepare(
        MAIN_MD,
        main_build,
        "R0_r as an Early Warning Indicator for Carbapenem-Resistant Klebsiella pneumoniae",
        MAIN_FIG_CAPTIONS,
        False,
    )
    prepare(SUPP_MD, supp_build, "Supplementary Appendix", SUPP_FIG_CAPTIONS, True)
    for md, out in [(main_build, MAIN_DOCX), (main_build, MAIN_PDF), (supp_build, SUPP_DOCX), (supp_build, SUPP_PDF)]:
        pandoc(md, out)
        print(f"Generated {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
