"""Checks of the compiled thesis (thesis/build/tesi.pdf). Exit code 0 only when every check passes.

    python scripts/check_thesis.py --dir thesis

Checks: pages of the body (<= 60, from the labels corpo:inizio and corpo:fine), fonts embedded, PDF/A-2b with veraPDF,
every citation in references.bib and every entry cited and verified (VERIFICATION.md), no placeholder left in the
sources, no undefined reference or citation in the LaTeX log. veraPDF missing counts as a failure: a PDF/A that was not
validated is reported as not verified, never as compliant.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_BODY_PAGES = 60
PLACEHOLDER = re.compile(r"\bTODO\b|(?i:da scrivere)|\?\?")


def body_pages(aux: str) -> int | None:
    """Pages between the labels corpo:inizio and corpo:fine (inclusive), from the .aux file."""
    pages = {}
    for key in ("corpo:inizio", "corpo:fine"):
        m = re.search(r"\\newlabel\{" + re.escape(key) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
        if not m:
            return None
        pages[key] = int(m[1])
    return pages["corpo:fine"] - pages["corpo:inizio"] + 1


def bib_keys(text: str) -> list[str]:
    return re.findall(r"^@\w+\{([^,\s]+),", text, flags=re.MULTILINE)


def cited_keys(bcf: str) -> set[str]:
    return set(re.findall(r"<bcf:citekey[^>]*>([^<]+)</bcf:citekey>", bcf)) - {"*"}


def verified_keys(text: str) -> set[str]:
    return set(re.findall(r"^## (\S+)", text, flags=re.MULTILINE))


def placeholders(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if PLACEHOLDER.search(line.split("%")[0] if not line.lstrip().startswith("%") else ""):
                hits.append(f"{path.name}:{n}: {line.strip()[:60]}")
    return hits


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="thesis")
    args = ap.parse_args()
    root = Path(args.dir)
    build, pdf = root / "build", root / "build" / "tesi.pdf"
    problems: list[str] = []
    if not pdf.exists():
        sys.exit(f"{pdf} does not exist: run `make pdf`")

    pages = int(re.search(r"Pages:\s+(\d+)", run(["pdfinfo", str(pdf)]))[1])
    body = body_pages((build / "tesi.aux").read_text())
    print(f"pages: {pages} in total, body {body if body is not None else 'unknown'} (limit {MAX_BODY_PAGES})")
    if body is None:
        problems.append("labels corpo:inizio / corpo:fine not found in tesi.aux")
    elif body > MAX_BODY_PAGES:
        problems.append(f"body has {body} pages, limit {MAX_BODY_PAGES}")

    fonts = run(["pdffonts", str(pdf)]).splitlines()[2:]
    loose = [line.split()[0] for line in fonts if len(line.split()) > 4 and "yes" not in line.split()[-6:-4] and " no " in line]
    print(f"fonts: {len(fonts)}, not embedded: {len(loose)}")
    problems += [f"font not embedded: {f}" for f in loose]

    if shutil.which("verapdf"):
        report = run(["verapdf", "--flavour", "2b", str(pdf)])
        ok = 'isCompliant="true"' in report
        print(f"PDF/A-2b (veraPDF): {'compliant' if ok else 'NOT compliant'}")
        if not ok:
            problems.append("veraPDF: not PDF/A-2b compliant")
    else:
        print("PDF/A-2b: NOT VERIFIED (veraPDF not found)")
        problems.append("veraPDF not found: PDF/A not verified")

    bib = bib_keys((root / "bib" / "references.bib").read_text())
    cited = cited_keys((build / "tesi.bcf").read_text()) if (build / "tesi.bcf").exists() else set()
    verified = verified_keys((root / "bib" / "VERIFICATION.md").read_text())
    print(f"bibliography: {len(bib)} entries, {len(cited)} cited, {len(verified)} verified")
    problems += [f"cited but not in references.bib: {k}" for k in sorted(cited - set(bib))]
    problems += [f"in references.bib but never cited: {k}" for k in sorted(set(bib) - cited)]
    problems += [f"in references.bib without a VERIFICATION.md section: {k}" for k in sorted(set(bib) - verified)]

    sources = [root / "tesi.tex", root / "metadata.tex", root / "abstract.tex", root / "tesi.xmpdata",
               *sorted((root / "chapters").glob("*.tex"))]
    left = placeholders(sources)
    print(f"placeholders left: {len(left)}")
    problems += [f"placeholder: {h}" for h in left[:20]]

    log = (build / "tesi.log").read_text(errors="replace")
    undefined = re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined", log)
    print(f"undefined references/citations: {len(undefined)}")
    problems += undefined[:10]

    print("\nRESULT:", "OK" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print(" -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
