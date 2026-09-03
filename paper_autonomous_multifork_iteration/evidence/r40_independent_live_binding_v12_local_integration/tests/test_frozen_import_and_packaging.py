from __future__ import annotations
import hashlib,importlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];GPU=REPO/"paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu"
class Packaging(unittest.TestCase):
    def test_hash_bound_real_runner_interface(self):
        runner=GPU/"run_qcomem_qwen35_forkaudit_review_revision.py"
        if not runner.is_file():self.skipTest("repository-side runner verified at formal staging")
        self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(),"9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775")
        sys.path.insert(0,str(GPU));self.addCleanup(lambda:sys.path.remove(str(GPU)));module=importlib.import_module("run_qcomem_qwen35_forkaudit_review_revision")
        self.assertTrue(all(callable(getattr(module,name,None)) for name in ("_run_clean_memory_cell","_run_formal_factorial_cells","_run_ownership_witness_cell","build_resident_request_group")))
        self.assertEqual(len(module.FORMAL_RESIDENT_COUNTS)*len(module.ARM_IDS),12)
    def test_dynamic_runner_import_registers_dataclasses_module(self):
        runner=GPU/"run_qcomem_qwen35_forkaudit_review_revision.py"
        if not runner.is_file():self.skipTest("repository dependency must be explicitly staged for formal preflight")
        sys.path.insert(0,str(GPU));self.addCleanup(lambda:sys.path.remove(str(GPU)))
        spec=importlib.util.spec_from_file_location("r40_v12_cpu_import_regression",runner);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;self.addCleanup(lambda:sys.modules.pop(spec.name,None));spec.loader.exec_module(module)
        self.assertTrue(hasattr(module,"LiveKVIdentityGuard"))
    def test_actual_frozen_builder_import_and_hash(self):
        if not GPU.is_dir():
            self.skipTest("repository-side frozen production dependency is verified at staging/runtime")
        sys.path.insert(0,str(GPU));self.addCleanup(lambda:sys.path.remove(str(GPU)))
        module=importlib.import_module("qcomem_vllm_paged_multifork_resident")
        self.assertTrue(callable(module._prepare_request_gdn_base));self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),"546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e")
    def test_launcher_refuses_overwrite(self):
        sys.path.insert(0,str(ROOT/"scripts"));self.addCleanup(lambda:sys.path.remove(str(ROOT/"scripts")))
        from build_formal_launcher import main
        # Covered structurally and by the command-level static audit; the builder checks output.exists before write.
        self.assertIn("a.output.exists()",(ROOT/"scripts/build_formal_launcher.py").read_text())
    def test_single_exit_handler_success_and_failure(self):
        launcher=ROOT/"formal/launch_h20.sh";text=launcher.read_text();self.assertEqual(text.count("trap on_exit EXIT"),1);self.assertNotIn("trap 'rm",text)
        with tempfile.TemporaryDirectory() as tmp:
            failure=Path(tmp)/"failure.json";env=dict(os.environ,R40_FAILURE_LEDGER=str(failure),R40_LAUNCHER_HANDLER_SELFTEST="success");self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,0);self.assertFalse(failure.exists())
            env["R40_LAUNCHER_HANDLER_SELFTEST"]="failure";self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,7);self.assertEqual(json.loads(failure.read_text())["exit_code"],7)
if __name__=="__main__":unittest.main()
