from __future__ import annotations
import argparse,hashlib,os,subprocess,sys,unittest
from pathlib import Path
RUNNER_SHA="9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
BUILDER_SHA="546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--repo-root",type=Path,required=True);a=p.parse_args()
    gpu=a.repo_root/"paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu";runner=gpu/"run_qcomem_qwen35_forkaudit_review_revision.py";builder=gpu/"qcomem_vllm_paged_multifork_resident.py"
    if sha(runner)!=RUNNER_SHA or sha(builder)!=BUILDER_SHA:raise SystemExit("explicit staged production dependency hash drift")
    os.environ["R40_V16_REPO_ROOT"]=str(a.repo_root.resolve(strict=True))
    suite=unittest.defaultTestLoader.discover(str(a.root/"tests"));result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise SystemExit("formal local-test preflight requires all tests successful with zero skip")
    subprocess.run([sys.executable,str(a.root/"scripts/static_audit.py")],check=True,cwd=a.root)
    return 0
if __name__=="__main__":raise SystemExit(main())
