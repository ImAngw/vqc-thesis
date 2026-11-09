import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_thesis", ROOT / "scripts" / "check_thesis.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def test_body_pages_counts_from_the_two_labels():
    aux = "\\newlabel{corpo:inizio}{{}{5}{}{section*.3}{}}\n\\newlabel{corpo:fine}{{}{57}{}{section*.9}{}}\n"
    assert ct.body_pages(aux) == 53
    assert ct.body_pages("\\newlabel{corpo:inizio}{{}{5}{}{x}{}}") is None


def test_bibliography_checks_find_missing_unused_and_unverified_keys():
    bib = ct.bib_keys("@article{a2020,\n title={x}\n}\n@inproceedings{b2021,\n title={y}\n}\n")
    cited = ct.cited_keys("<bcf:citekey order=\"1\">a2020</bcf:citekey><bcf:citekey>c2022</bcf:citekey><bcf:citekey>*</bcf:citekey>")
    verified = ct.verified_keys("# T\n## a2020\nsource\n")
    assert bib == ["a2020", "b2021"] and cited == {"a2020", "c2022"}
    assert cited - set(bib) == {"c2022"}  # cited but missing
    assert set(bib) - cited == {"b2021"}  # never cited
    assert set(bib) - verified == {"b2021"}  # no verification


def test_placeholders_ignore_comments_and_flag_todo(tmp_path):
    f = tmp_path / "c.tex"
    f.write_text("% TODO in a comment line\nTesto normale % TODO dopo il testo\nDa scrivere.\nRiferimento ??\n")
    hits = ct.placeholders([f])
    assert len(hits) == 2 and "c.tex:3" in hits[0] and "c.tex:4" in hits[1]
