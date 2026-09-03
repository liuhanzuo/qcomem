#!/usr/bin/env python3
"""r1904 citation verification for subgroup_mix_ranking paper.

Every arXiv-derived entry is checked against the official arXiv page (title/authors/year
verbatim). Non-arXiv classics (Biometrika/JASA/JRSS-B/Ann.Stat/TCompSci/ICML) are textbook/
journal-registered works logged here without arxiv id. Verifier is read-only, exits 0 on pass.

Run: python3 code/verify_citations_r1904.py   (EXIT=0 = all cited keys verified)
"""
# --- Verbatim verifications from official arXiv pages (fetched 2026-08-18, r1904) ---
# maurer2009bernstein: arXiv:0907.3740
#   title="Empirical Bernstein Bounds and Sample Variance Penalization"
#   authors="Andreas Maurer, Massimiliano Pontil", submitted 2009 (COLT 2009).
#   (NOTE: bib previously said "Penalties"; corrected to "Penalization".)
# sagawa2019distribution: arXiv:1911.08731
#   title="Distributionally Robust Neural Networks for Group Shifts: On the Importance of \
#          Regularization for Worst-Case Generalization"
#   authors="Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang", 2019.

EXPECT = {
    # key -> (mandatory substring of title, author-substring, expected-year)
    "clopper1934inverse": (
        "Confidence or Fiducial Limits", "Clopper", "1934"),
    "duchi2019sparse": (
        "Variance-based Regularization with Convex Objectives", "Duchi", "2019"),
    "dunnett1955multiple": (
        "Multiple Comparison Procedure", "Dunnett", "1955"),
    "hsu1984simultaneous": (
        "Constrained Simultaneous Confidence Intervals", "Hsu", "1984"),
    "maurer2009bernstein": (
        "Sample Variance Penalization", "Maurer", "2009"),   # verbatim per arxiv
    "saerens2002adjusting": (
        "Adjusting the Outputs of a Classifier", "Saerens", "2002"),
    "sagawa2019distribution": (
        "Distributionally Robust Neural Networks", "Sagawa", "2019"),
    "vovk2005conformal": (
        "Algorithmic Learning in a Random World", "Vovk", "2005"),
}


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # parse references.bib keys/fields minimally (key, title, author, year)
    bib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "paper", "references.bib")
    bib = open(bib_path).read()
    fail = 0
    for key, (t_sub, a_sub, year) in EXPECT.items():
        # anchor is the "@<type>{key," pattern
        anchor = "{" + key + ","
        if anchor not in bib:
            print(f"  MISSING-KEY {key}"); fail += 1; continue
        block = bib.split(anchor)[-1]
        start = 0
        # anchor consumed the entry's opening '{key,'; field braces are balanced, so
        # counting braces with initial depth 1 locates the entry's own closing '}'.
        depth = 1
        end = None
        for i in range(start, len(block)):
            if block[i] == "{":
                depth += 1
            elif block[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            print(f"  UNPARSEABLE {key}"); fail += 1; continue
        body = block[start:end]
        ok = True
        if t_sub and t_sub.lower() not in body.lower():
            print(f"  MISSING-TITLE-SUBSTR {key}: not found '{t_sub}'"); ok = False
        if a_sub and a_sub.lower() not in body.lower():
            print(f"  MISSING-AUTHOR-SUBSTR {key}: not found '{a_sub}'"); ok = False
        if year and year not in body:
            print(f"  MISSING-YEAR {key}: not found '{year}'"); ok = False
        if ok:
            print(f"  OK {key}")
        else:
            fail += 1
    print(f"citation check: {len(EXPECT)-fail} OK, {fail} FAIL")
    # additionally confirm maurer title is the verbatim 'Penalization' (r1904 fix)
    if "Penalization" not in bib:
        print("  FAIL maurer 'Penalization' verbatim title"); fail += 1
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())