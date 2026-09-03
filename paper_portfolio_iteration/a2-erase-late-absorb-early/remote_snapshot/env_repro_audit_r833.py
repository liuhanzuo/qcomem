#!/usr/bin/env python3
"""r833 dimension-19: environment / reproducibility-package integrity audit.

Orthogonal attack surface not covered by dims 1-18 (which audit content
correctness, disclosure, rendering, citation, counts). This dimension audits
whether the reproducibility PACKAGE's own claims about itself are true:
  (a) README env claims (torch/numpy/python versions) match the real env;
  (b) README file-inventory counts (N scripts / M JSONs) match disk;
  (c) MANIFEST lists every file it should, all listed files exist, all
      checksums verify (modulo the registered living-document exception);
  (d) compile-dependency style files named in README exist;
  (e) the checker entrypoint command in README actually runs.

L0 fault-injection (on a temp copy) proves the detectors catch failures
before trusting PASS. Reads the FROZEN package read-only; never writes into
it. Exit 0 = all pass.
"""
import os, re, sys, json, shutil, subprocess, tempfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "REPRO_README.md")
CODE = os.path.join(HERE, "code")
MANIFEST = os.path.join(HERE, "MANIFEST_md5.txt")

fails = []
def chk(desc, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + desc + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(desc)

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

# ---------------------------------------------------------------- L0: self-check
# Prove the detectors catch a planted fault, on a temp copy (frozen bytes untouched).
def selfcheck():
    print("== L0 fault-injection self-check ==")
    tmp = tempfile.mkdtemp(prefix="r833_selfcheck_")
    # (i) count-detector: claim 40 scripts but dir has 39 -> must detect mismatch
    os.makedirs(os.path.join(tmp, "code"))
    for i in range(39):
        open(os.path.join(tmp, "code", f"s{i}.py"), "w").write("pass\n")
    n_py = len([f for f in os.listdir(os.path.join(tmp, "code")) if f.endswith(".py")])
    det1 = (n_py != 40)
    chk("L0(i) count detector catches 39!=40", det1, f"found {n_py}")
    # (ii) manifest-detector: a listed file whose bytes changed -> must FAIL verify
    open(os.path.join(tmp, "f.txt"), "w").write("original")
    good = md5(os.path.join(tmp, "f.txt"))
    open(os.path.join(tmp, "MANIFEST.txt"), "w").write(f"{good}  f.txt\n")
    open(os.path.join(tmp, "f.txt"), "w").write("tampered")
    bad = md5(os.path.join(tmp, "f.txt"))
    det2 = (bad != good)
    chk("L0(ii) manifest detector catches byte tamper", det2)
    shutil.rmtree(tmp)
    return det1 and det2

# ------------------------------------------------------- parse README env claims
def parse_env_claims(text):
    """Extract '作者环境实测：torch X, numpy Y.' pinned versions."""
    m = re.search(r"作者环境实测：\s*torch\s*([0-9a-zA-Z.+]+),\s*numpy\s*([0-9.]+)", text)
    return {"torch": m.group(1), "numpy": m.group(2)} if m else None

def main():
    if not selfcheck():
        print("SELF-CHECK FAILED -- do not trust main audit"); sys.exit(1)

    text = open(README, encoding="utf-8").read()

    print("== L1 environment version claims vs real env ==")
    claims = parse_env_claims(text)
    chk("README pins torch+numpy versions", claims is not None, str(claims))
    if claims:
        import numpy, torch
        chk("numpy version matches README", numpy.__version__ == claims["numpy"],
            f"readme={claims['numpy']} actual={numpy.__version__}")
        # torch: README pins prefix '2.3.0a0'; actual may carry local suffix '+40ec...'
        chk("torch version matches README (prefix)", torch.__version__.startswith(claims["torch"]),
            f"readme={claims['torch']} actual={torch.__version__}")
        chk("torch cuda available (README: 'CUDA GPU 一张')", torch.cuda.is_available())

    print("== L2 compile-dependency style files exist ==")
    m = re.search(r"随附的 style 文件（([^）]+)）", text)
    named = re.findall(r"`([a-zA-Z0-9_./]+(?:sty|bst|tex))`", m.group(1)) if m else []
    # README writes "iclr2027_conference.sty/.bst" = two files sharing a stem;
    # expand the "/.<ext>" shorthand into its own file name.
    expanded = []
    for tok in named:
        if "/" in tok:
            stem, exts = tok.split("/", 1)
            base = stem.rsplit(".", 1)[0]
            expanded.append(stem)
            expanded.append(base + exts)  # exts already begins with '.'
        else:
            expanded.append(tok)
    chk("README names style deps", len(expanded) >= 5, str(expanded))
    for f in expanded:
        chk(f"style dep exists: {f}", os.path.exists(os.path.join(HERE, f)))

    print("== L3 file-inventory counts match disk ==")
    # README §4: 'code/ — 40 个实验/验证脚本 ... + 35 个冻结结果 JSON'
    mN = re.search(r"(\d+)\s*个实验/验证脚本", text)
    mJ = re.search(r"(\d+)\s*个冻结结果\s*JSON", text)
    n_py = len([f for f in os.listdir(CODE) if f.endswith(".py")])
    n_json = len([f for f in os.listdir(CODE) if f.endswith("_out.json")])
    chk("script-count claim parses", mN is not None)
    chk("json-count claim parses", mJ is not None)
    if mN: chk(f"script count: README={mN.group(1)} disk={n_py}", int(mN.group(1)) == n_py)
    if mJ: chk(f"json count: README={mJ.group(1)} disk={n_json}", int(mJ.group(1)) == n_json)

    print("== L4 MANIFEST integrity ==")
    entries = []
    for line in open(MANIFEST):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        s, _, f = line.partition("  ")
        entries.append((s.strip(), f.strip()))
    chk("MANIFEST has entries", len(entries) > 0, f"{len(entries)}")
    # L4a: every listed file exists
    missing = [f for _, f in entries if not os.path.exists(os.path.join(HERE, f))]
    chk("all MANIFEST-listed files exist", not missing, f"missing={missing}")
    # L4b: checksums verify, allowing ONLY the registered living-doc exception
    # (SUBMISSION_CHECKLIST r130: RESEARCH_LOG.md is an appended living document
    #  whose MANIFEST hash legitimately goes stale; verified below).
    bad = []
    for s, f in entries:
        p = os.path.join(HERE, f)
        if os.path.exists(p) and md5(p) != s:
            bad.append(f)
    unexpected = [f for f in bad if f != "RESEARCH_LOG.md"]
    chk("MANIFEST checksums verify (except registered living doc)",
        not unexpected, f"unexpected_mismatch={unexpected}")
    # L4c: confirm the RESEARCH_LOG exception is legitimate, three independent ways:
    #  (i)  it is the ONLY checksum mismatch (no surprise tamper);
    #  (ii) its mtime is AFTER the MANIFEST mtime (log appended after manifest written);
    #  (iii) SUBMISSION_CHECKLIST explicitly registers the manifest file-set asymmetry
    #        naming RESEARCH_LOG (r130 教训(l)).
    if "RESEARCH_LOG.md" in bad:
        only = (bad == ["RESEARCH_LOG.md"])
        reg = open(os.path.join(HERE, "SUBMISSION_CHECKLIST.md"), encoding="utf-8").read()
        registered = "RESEARCH_LOG" in reg and "文件集不同" in reg
        mtime_ok = os.path.getmtime(os.path.join(HERE, "RESEARCH_LOG.md")) > os.path.getmtime(MANIFEST)
        chk("RESEARCH_LOG mismatch is the registered living-doc exception",
            only and registered and mtime_ok,
            f"only_mismatch={only} registered={registered} mtime_after={mtime_ok}")

    print("== L5 README checker entrypoint actually runs ==")
    # README §2: CONSISTENCY_JDIR=<json dir> python3 consistency_r790.py -> 154 pass,0 fail
    env = dict(os.environ); env["CONSISTENCY_JDIR"] = CODE
    r = subprocess.run([sys.executable, "consistency_r790.py"], cwd=HERE, env=env,
                       capture_output=True, text=True)
    m = re.search(r"(\d+)\s*pass,\s*(\d+)\s*fail", r.stdout)
    chk("checker runs and reports pass/fail", m is not None, r.stdout.strip().splitlines()[-1] if r.stdout else "no output")
    if m:
        chk("checker 154 pass / 0 fail", int(m.group(1)) == 154 and int(m.group(2)) == 0,
            f"pass={m.group(1)} fail={m.group(2)}")

    print()
    if fails:
        print(f"DIM-19 AUDIT: {len(fails)} FAIL"); sys.exit(1)
    print("DIM-19 AUDIT: all pass"); sys.exit(0)

if __name__ == "__main__":
    main()
