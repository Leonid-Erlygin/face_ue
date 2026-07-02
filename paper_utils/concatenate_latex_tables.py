#!/usr/bin/env python3
"""
Concatenate all .tex tables from the `tex/` subdirectories of the given
output directories into a single combined .tex file.
"""

from pathlib import Path

# Base directory containing the 4 table dirs (adjust if needed).
BASE_DIR = Path("/app/outputs")

SOURCE_DIRS = [
    "latex_tables_new_paper_bio",
    "latex_tables_new_paper_bio_diagnostics",
    "latex_tables_new_paper_text",
    "latex_tables_new_paper_text_diagnostics",
]

OUTPUT_FILE = BASE_DIR / "all_tables_combined.tex"


def collect_tex_files(base_dir: Path, source_dirs):
    """Return a sorted list of (relative_label, path) for every .tex file found."""
    tex_files = []
    for d in source_dirs:
        tex_dir = base_dir / d / "tex"
        if not tex_dir.is_dir():
            print(f"[warning] missing tex dir: {tex_dir}")
            continue
        for tex_path in sorted(tex_dir.glob("*.tex")):
            label = f"{d}/{tex_path.name}"
            tex_files.append((label, tex_path))
    return tex_files


def main():
    tex_files = collect_tex_files(BASE_DIR, SOURCE_DIRS)

    if not tex_files:
        print("[error] no .tex files found.")
        return

    lines = [
        "% ============================================================",
        "% Auto-generated combined tables file",
        f"% Total tables: {len(tex_files)}",
        "% ============================================================",
        "",
    ]

    for label, tex_path in tex_files:
        try:
            content = tex_path.read_text(encoding="utf-8").strip()
        except Exception as e:  # noqa: BLE001
            print(f"[warning] could not read {tex_path}: {e}")
            continue

        lines.append("% " + "-" * 60)
        lines.append(f"% Source: {label}")
        lines.append("% " + "-" * 60)
        lines.append(content)
        lines.append("")  # blank line between tables
        lines.append("\\clearpage")
        lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {len(tex_files)} tables to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()