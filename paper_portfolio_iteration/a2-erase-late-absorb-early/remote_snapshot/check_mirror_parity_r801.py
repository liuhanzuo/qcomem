#!/usr/bin/env python3
"""r801: 镜像完整性机器核对 — 权威端每个常规文件在镜像必须存在且逐字节一致。
教训g收尾四步只核验 MANIFEST 登记文件，本脚本核全目录(含未登记 build log)。"""
import os, sys, filecmp

AUTH = "/newcpfs/user/qixuan1/01_p5/run/iclr27_theory_k3_20260806_r1/paper/A2_lrphase"
MIR  = "/newcpfs/user/qixuan1/01_p5/run/iclr27_theory_k3_20260806_r1/agents/A2/workspace/paper"

def check(sub=""):
    a, m = os.path.join(AUTH, sub), os.path.join(MIR, sub)
    missing, stale = [], []
    for f in sorted(os.listdir(a)):
        af = os.path.join(a, f)
        if not os.path.isfile(af):
            continue
        mf = os.path.join(m, f)
        if not os.path.isfile(mf):
            missing.append(os.path.join(sub, f))
        elif not filecmp.cmp(af, mf, shallow=False):
            stale.append(os.path.join(sub, f))
    return missing, stale

def main():
    miss, stale = check("")
    cm, cs = check("code")
    miss += cm; stale += cs
    for f in miss:   print(f"MISSING in mirror: {f}")
    for f in stale:  print(f"STALE in mirror:   {f}")
    n = sum(1 for sub in ("", "code")
            for f in os.listdir(os.path.join(AUTH, sub))
            if os.path.isfile(os.path.join(AUTH, sub, f)))
    print(f"=== mirror parity: {n - len(miss) - len(stale)}/{n} OK, "
          f"missing={len(miss)}, stale={len(stale)} ===")
    sys.exit(1 if (miss or stale) else 0)

if __name__ == "__main__":
    main()
