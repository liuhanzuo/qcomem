#!/usr/bin/env python3
"""arxiv_drift_r802.py — EraseLateAbsorbEarly 投稿日终检 arXiv 漂移复跑脚本（预注册，A2 r802）

预注册条款（先于首次运行写死，之后不得看结果再改）：
- 漂移锚 ANCHOR=2026-08-06（run 起点，与 A5 SUFFCRIT 漂移锚同制）；凡 published >= 锚点的
  检索命中条目记为"新进"，需逐条人工判读是否覆盖本线对象。
- 关线规则（依 2026-08-07 增补）：仅当某条目同时覆盖 (1) LR相位/暴露时刻数据价值分解、
  (2) 吸收/擦除不同构、(3) 机制解释+形式化、(4) 相同主要结论，才构成关线理由；
  1-2 篇相似不关线，记 FOLLOWUP_DELTA 待核条目。
- 探针：KNOWN_IDS 为近邻表 15 篇中 14 篇有确定 arXiv ID 者（Datamodels/MAGIC/REPLAY 无
  稳定一手 ID 核验记录，不硬编）；探针全部解析成功 = API 链路正常（否则检索空结果不可信）。
- 输出 arxiv_drift_r802_out.json：每查询 totalResults + 新进条目表 + 探针解析表 + 错误表。
  复跑性核验方式：同日重跑本脚本，totalResults 与探针表应逐位一致（检索结果可增不可减）。
"""
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

ANCHOR = "2026-08-06"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
OPS = {"o": "http://a9.com/-/spec/opensearch/1.1/"}
API = "http://export.arxiv.org/api/query"
SLEEP = 3.5  # arXiv 官方要求 >=3s

KNOWN_IDS = [  # 近邻表 14 篇（r774/r775/r777/r789 一手核验记录）
    "2505.22509",  # Xie differentiable stopping time（r775 修正，不占本线对象）
    "2603.19688",  # DataProphet
    "2303.14186",  # TRAK
    "2002.08484",  # TracIn
    "1812.05159",  # Toneva forgetting
    "1912.03817",  # SISA
    "2012.07805",  # Carlini extraction
    "2404.06395",  # MiniCPM WSD（§4.4 自认衰减期机制 open）
    "1710.06451",  # Smith & Le 噪声尺度
    "1704.04289",  # Mandt SGD-Bayes
    "2101.12176",  # Smith/Dherin 隐式正则
    "2011.09468",  # Pezeshki gradient starvation
    "1908.00045",  # Safran-Shamir RR lower bound
    "1903.01463",  # Nagaraj without-replacement
    "2006.05988",  # Mishchenko RR analysis
]

QUERIES = [  # 面向本线四维对象的漂移检索（互补覆盖，不硬上限）
    ("Q1", 'all:"data valuation" AND all:"learning rate schedule"'),
    ("Q2", 'all:"data attribution" AND all:"training phase"'),
    ("Q3", 'all:"learning rate decay" AND all:"forgetting" AND cat:cs.LG'),
    ("Q4", 'abs:"phase-dependent" AND abs:"data value"'),
    ("Q5", 'abs:"exposure time" AND abs:"training dynamics"'),
    ("Q6", 'all:"machine unlearning" AND all:"learning rate decay"'),
    ("Q7", 'all:"data ordering" AND all:"learning rate" AND abs:"value"'),
]


def fetch(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "a2-lrphase-drift/1.0 (mailto:none)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", ATOM):
        out.append({
            "id": e.find("a:id", ATOM).text.rsplit("/", 1)[-1],
            "published": e.find("a:published", ATOM).text[:10],
            "title": " ".join(e.find("a:title", ATOM).text.split()),
        })
    total = root.find("o:totalResults", OPS)
    return (int(total.text) if total is not None else None), out


def main():
    res = {"anchor": ANCHOR, "run": "r802", "queries": {}, "id_probe": {}, "errors": []}
    for name, q in QUERIES:
        try:
            total, ents = entries(fetch({"search_query": q, "sortBy": "submittedDate",
                                         "sortOrder": "descending", "max_results": 50}))
            new = [e for e in ents if e["published"] >= ANCHOR]
            res["queries"][name] = {"q": q, "totalResults": total, "n_returned": len(ents),
                                    "n_new_since_anchor": len(new), "new": new}
        except Exception as ex:
            res["queries"][name] = {"q": q, "error": repr(ex)}
            res["errors"].append(f"{name}: {ex!r}")
        time.sleep(SLEEP)
    try:
        _, ents = entries(fetch({"id_list": ",".join(KNOWN_IDS), "max_results": 50}))
        got = {e["id"].split("v")[0] for e in ents}
        res["id_probe"] = {"expected": len(KNOWN_IDS), "resolved": len(got & set(KNOWN_IDS)),
                           "missing": sorted(set(KNOWN_IDS) - got),
                           "titles": {e["id"].split("v")[0]: e["title"] for e in ents}}
    except Exception as ex:
        res["id_probe"] = {"error": repr(ex)}
        res["errors"].append(f"id_probe: {ex!r}")
    out_path = "arxiv_drift_r802_out.json"
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    # 终端摘要
    print("anchor:", ANCHOR)
    for n, r in res["queries"].items():
        if "error" in r:
            print(f"{n}: ERROR {r['error']}")
        else:
            print(f"{n}: total={r['totalResults']} new>={ANCHOR}: {r['n_new_since_anchor']}")
            for e in r["new"]:
                print(f"   NEW {e['id']} {e['published']} {e['title'][:110]}")
    p = res["id_probe"]
    print("id_probe:", p if "error" in p else f"{p['resolved']}/{p['expected']} resolved, missing={p['missing']}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
