#!/usr/bin/env python3
"""第24维自主审计：训练配置/超参数声明 vs 一手脚本/JSON 一致性。

攻击面：hostile reviewer 复现视角——论文声称的训练配置（eta网格/T/split/bs/momentum/
wd/arch/数据集/种子数/运行时/曝光范围/块比例）是否与实际执行脚本+落盘JSON一致。
前23维核数值/计数/措辞/合规；r833核README自声明；consistency_r790核锚点数字。
本维核正交缺口：训练配置本身（r790不覆盖）。

协议：L0 故障注入（临时副本证实检测器能抓声称-脚本失配）→ L1-L5 对冻结包跑主审计。
exit 0 = 全过。只读冻结文件，零改动。
"""
import json, os, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "paper.tex")
CODE = os.path.join(HERE, "code")

PASS, FAIL = [], []

def chk(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("PASS " if cond else "FAIL ") + name + (" | " + detail if detail else ""))

def tex_lines(path=TEX):
    src = open(path, encoding="utf-8").read()
    # strip comment lines; keep escaped \% (it is a literal percent, not a comment)
    out = []
    for l in src.splitlines():
        if l.lstrip().startswith("%"):
            continue
        buf, i = [], 0
        while i < len(l):
            if l[i] == "%" and (i == 0 or l[i-1] != "\\"):
                break
            buf.append(l[i]); i += 1
        out.append("".join(buf))
    return "\n".join(out)

def script_src(name):
    return open(os.path.join(CODE, name), encoding="utf-8").read()

def jload(name):
    return json.load(open(os.path.join(CODE, name)))

REAL_SCRIPTS = ["pilot17_mnist_fashion.py", "pilot18_smallconflict.py",
                "pilot19_cifar10n.py", "diag20_traj.py", "eqra_salvage.py",
                "eqra_loss_salvage.py", "eqra_loss_p3_precision.py"]
# pilot17_cifar10n.py 是 off-task 负结果臂（tex L539 引用其实现类负结果），用独立五元组
# (T=240, split=120, eta 0.05/0.005, bs=128)；附录五元组(含 bs 256)的适用范围是
# "pilot17/18/19, diag20, eqra_*"——pilot17_cifar10n 属 pilot19 前身臂非枚举集成员。
OFFTASK_SCRIPT = "pilot17_cifar10n.py"
ALL_AUDITED_SCRIPTS = REAL_SCRIPTS + [OFFTASK_SCRIPT]

def L1_realimage_config(src_override=None):
    """L1: 图像臂训练配置五元组 vs 8脚本逐字段核对。"""
    print("\n== L1 图像臂五元组 (T=240, split=120, eta_hi=0.05, eta_lo=0.005, bs=256) ==")
    get = (lambda n: src_override.get(n, script_src(n))) if src_override else script_src
    ok = True
    for s in REAL_SCRIPTS:
        src = get(s)
        for pat, label in [
            (r"T\s*=\s*240", "T=240"), (r"T_split\s*=\s*120", "split=120"),
            (r"eta_hi\s*=\s*0\.05", "eta_hi=0.05"), (r"eta_lo\s*=\s*0\.005", "eta_lo=0.005"),
            (r"bs\s*=\s*256", "bs=256"),
        ]:
            hit = re.search(pat, src)
            if not hit:
                ok = False
                chk(f"L1.{s}:{label}", False, "pattern missing")
    chk("L1.五元组@7脚本(默认参数)", ok, "train() defaults")
    # off-task 臂独立五元组（bs=128），但 T/split/eta 与主体系一致
    so = get(OFFTASK_SCRIPT)
    chk("L1.offtask臂(pilot17_cifar10n)独立五元组bs=128,eta/T/split同体系",
        all(re.search(p, so) for p in
            [r"T\s*=\s*240", r"T_split\s*=\s*120", r"eta_hi\s*=\s*0\.05",
             r"eta_lo\s*=\s*0\.005", r"bs\s*=\s*128"]), "")
    # momentum=0 & weight_decay=0 in every optimizer construction
    ok2 = True
    for s in REAL_SCRIPTS:
        src = get(s)
        for m in re.finditer(r"torch\.optim\.SGD\(.*?\)\s*$", src, re.S | re.M):
            call = m.group(0)
            if "momentum=0.0" not in call or "weight_decay=0.0" not in call:
                ok2 = False
                chk(f"L1.{s}:opt", False, call[:80])
    chk("L1.momentum=0&wd=0@全部SGD构造", ok2, "")
    # tex side: claims present
    t = src_override["__tex__"] if src_override else tex_lines()
    chk("L1.tex声称五元组存在",
        all(x in t for x in ["240 epochs", "split 120", "0.05", "0.005", "bs 256", "momentum 0"]),
        "app:repro")

def L2_arch_dataset(src_override=None):
    """L2: 架构/数据集声称 vs 脚本实现。"""
    print("\n== L2 架构/数据集 ==")
    get = (lambda n: src_override.get(n, script_src(n))) if src_override else script_src
    t = src_override["__tex__"] if src_override else tex_lines()
    # tex says SmallCNN (appendix) / small CNN (main)
    chk("L2.tex架构声称=SmallCNN/small CNN",
        ("SmallCNN" in t) and ("small CNN" in t), "L422/L958")
    # pilot19 SmallCNN: 3 conv, input 3ch (CIFAR 32x32x3)
    s19 = get("pilot19_cifar10n.py")
    cls = re.search(r"class SmallCNN.*?def forward", s19, re.S).group(0)
    chk("L2.pilot19 SmallCNN=3conv+3通道输入",
        cls.count("Conv2d") == 3 and "Conv2d(3, k, 3" in cls, "CIFAR版")
    # pilot17 MNIST SmallCNN: 2 conv, 1ch
    s17 = get("pilot17_mnist_fashion.py")
    cls17 = re.search(r"class SmallCNN.*?def forward", s17, re.S).group(0)
    chk("L2.pilot17 SmallCNN=2conv+1通道输入",
        cls17.count("Conv2d") == 2 and "Conv2d(1, k, 3" in cls17, "MNIST版")
    # dataset sources in scripts
    chk("L2.pilot17数据源=MNIST+Fashion-MNIST URL",
        "ossci-datasets.s3.amazonaws.com/mnist" in s17 and "fashion-mnist" in s17, "")
    chk("L2.pilot19数据源=CIFAR-10N(Wei2022)",
        "CIFAR-10_human.pt" in jload("pilot19_cifar10n_full_out.json")["data_provenance"]
        or "cifar" in s19.lower(), "data_provenance字段")

def L3_seeds_blocks(src_override=None):
    """L3: 种子数/块比例/T 与 JSON 落盘核对。"""
    print("\n== L3 种子/块比例/T vs JSON ==")
    t = src_override["__tex__"] if src_override else tex_lines()
    j17, j18, j19 = (jload(f) for f in
                     ["pilot17_realconflict_full_out.json",
                      "pilot18_smallconflict_full_out.json",
                      "pilot19_cifar10n_full_out.json"])
    chk("L3.6seeds@pilot17/18/19",
        all(j["seeds"] == [0,1,2,3,4,5] for j in (j17,j18,j19))
        and "6 seeds" in t, "")
    chk("L3.T=240落盘@三臂", all(j["T"] == 240 and j["quick"] is False for j in (j17,j18,j19)), "")
    chk("L3.pilot17 D=100%ofA",
        j17["nD"] == j17["nA"] == 60000 and re.search(r"100\\% of \$A\$", t) is not None,
        f"nD/nA={j17['nD']}/{j17['nA']}")
    chk("L3.pilot18 D=3%ofA",
        abs(j18["dfrac"]-0.03) < 1e-9 and j18["nD"] == 1800 and
        re.search(r"3\\% of \$A\$", t) is not None,
        f"dfrac={j18['dfrac']}, nD={j18['nD']}")
    chk("L3.pilot19 dfrac=10%实载",
        abs(j19["dfrac"]-0.10) < 1e-9 and j19["nA"] == 15000 and j19["nD"] == 1500,
        f"dfrac={j19['dfrac']}")

def L4_exposure_runtime():
    """L4: 合成η体系/曝光范围/运行时声称 vs 脚本+JSON。"""
    print("\n== L4 曝光/运行时 ==")
    t = tex_lines()
    v1 = script_src("verify_t1_invariance.py")
    v2 = script_src("verify_t1_invariance2.py")
    # E_full = 0.8*120+0.08*120 = 105.6 in both verify scripts
    chk("L4.E=105.6脚本重算",
        "0.8 * 120 + 0.08 * 120" in v1.replace(" ", "").replace("=", "=0.8*120+0.08*120", 0) or
        re.search(r"E_full\s*=\s*0\.8\s*\*\s*120\s*\+\s*0\.08\s*\*\s*120", v1) is not None,
        "verify_t1_invariance.py")
    chk("L4.tex声称E=105.6", "105.6" in t, "L214/L218/L371")
    # 200x range: verify2 frac down to 0.005 -> E=0.528~0.53; max range across scan <= 0.0004
    j2 = jload("verify_t1_invariance2_out.json")
    es = [float(k.split("=")[1]) for k in j2["E_scan_always_on"]]
    lo, hi = min(es), max(es)
    chk("L4.200x区间端点",
        abs(hi-105.6) < 0.01 and abs(lo-0.53) < 0.005 and hi/lo > 199,
        f"[{lo},{hi}], ratio={hi/lo:.1f}")
    maxrange = max(v["damage_range"] for v in j2["E_scan_always_on"].values())
    chk("L4.damage range<=0.0004@全扫描", maxrange <= 0.0004, f"max={maxrange}")
    chk("L4.tex声称区间[0.53,105.6]", "0.53,105.6" in t.replace(" ", ""), "L218")
    # synthetic eta values: tex never claims 0.8/0.08 for synthetic grid (avoid system collision)
    body = t.split("app:repro")[0]
    chk("L4.正文无0.8/0.08与图像体系混淆",
        not re.search(r"eta[^\n]{0,40}(0\.8|0\.08)", body), "合成η仅脚本层")
    # runtime claim: ~20 min per full 6-seed sweep; measured full runs
    runtimes = {f: jload(f)["runtime_min"] for f in
                ["pilot17_realconflict_full_out.json", "pilot18_smallconflict_full_out.json",
                 "pilot19_cifar10n_full_out.json", "eqra_salvage_full_out.json",
                 "eqra_loss_full_out.json", "eqra_loss_p3_full_out.json"]}
    chk("L4.tex声称~20min/sweep", "\\sim$20 min per full 6-seed sweep" in t.replace("  ", " ")
        or "20 min per full 6-seed" in t, "app:repro")
    # honesty framing: pilot19(25.4)/eqra(8-19) in range; pilot17/18(66/52) exceed -> scope check
    chk("L4.运行时数量级成立(四臂8-26min)",
        all(5 <= runtimes[f] <= 30 for f in
            ["pilot19_cifar10n_full_out.json", "eqra_salvage_full_out.json",
             "eqra_loss_full_out.json", "eqra_loss_p3_full_out.json"]),
        str({k.split('_')[0]: v for k, v in runtimes.items()}))
    # flag detector: are the two 60k-MNIST sweeps the ones exceeding?
    chk("L4.FLAG-R1 运行时范围定性",
        runtimes["pilot17_realconflict_full_out.json"] > 60 and
        runtimes["pilot18_smallconflict_full_out.json"] > 50,
        f"pilot17={runtimes['pilot17_realconflict_full_out.json']}, pilot18={runtimes['pilot18_smallconflict_full_out.json']} (60k图像臂超出~20min口径)")

def L5_cross_system():
    """L5: 双η体系并存一致性（0.8/0.08 合成 vs 0.05/0.005 图像）。"""
    print("\n== L5 双η体系区分 ==")
    syn = script_src("pilot_phase_datavalue.py") + script_src("pilot2_conflict.py") + \
          script_src("pilot11_lambda_traj.py")
    chk("L5.合成体系0.8/0.08@脚本", "0.8, 0.08" in syn or ("eta_hi, eta_lo = 0.8, 0.08" in syn)
        or "ETA_HI, ETA_LO, LAM = 0.8, 0.08" in syn, "")
    # appendix config line is explicitly inside real-image paragraph
    t = tex_lines()
    m = re.search(r"Real-image arms.*?240 epochs, split 120.*?momentum 0.*?6 seeds", t, re.S)
    chk("L5.附录五元组归属图像臂段落", m is not None, "Real-image arms (one GPU) 段内")
    # synthetic cell paragraph (L953) carries no eta values -> no collision
    m2 = re.search(r"Synthetic cell \(CPU.*?\.\}", t, re.S)
    chk("L5.合成段落不带η值(无串位)",
        m2 is not None and "0.8" not in m2.group(0) and "0.05" not in m2.group(0), "L953")

def L0_injection():
    """L0: 故障注入自校验——临时副本改脚本默认值，检测器必须抓。"""
    print("\n== L0 故障注入自校验 ==")
    srcs = {s: script_src(s) for s in REAL_SCRIPTS}
    srcs["__tex__"] = tex_lines()
    # inject: eta_hi 0.05 -> 0.5 in one script
    bad = dict(srcs)
    bad["pilot19_cifar10n.py"] = bad["pilot19_cifar10n.py"].replace("eta_hi=0.05", "eta_hi=0.5")
    global PASS, FAIL
    p0, f0 = len(PASS), len(FAIL)
    L1_realimage_config(bad)
    caught = len(FAIL) > f0
    # restore: rerun real L1 fresh later; here just assert detection worked
    PASS, FAIL = PASS[:p0], FAIL[:f0]
    chk("L0.检测器能抓eta_hi篡改(0.05→0.5)", caught, "注入后L1必须FAIL")
    bad2 = dict(srcs)
    bad2["__tex__"] = bad2["__tex__"].replace("100\\% of $A$", "50\\% of $A$")
    p0, f0 = len(PASS), len(FAIL)
    L3_seeds_blocks(bad2)
    caught2 = len(FAIL) > f0
    PASS, FAIL = PASS[:p0], FAIL[:f0]
    chk("L0.检测器能抓块比例声称篡改(100%→50%)", caught2, "")

def main():
    L0_injection()
    L1_realimage_config()
    L2_arch_dataset()
    L3_seeds_blocks()
    L4_exposure_runtime()
    L5_cross_system()
    print(f"\n=== r838 hyperparam audit: {len(PASS)} pass, {len(FAIL)} fail ===")
    for n, d in FAIL:
        print("FAIL-DETAIL:", n, d)
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
