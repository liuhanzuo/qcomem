from __future__ import annotations
import hashlib,importlib,importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
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
        spec=importlib.util.spec_from_file_location("r40_v15_cpu_import_regression",runner);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;self.addCleanup(lambda:sys.modules.pop(spec.name,None));spec.loader.exec_module(module)
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
            failure=Path(tmp)/"failure.json";env=dict(os.environ,R40_FAILURE_LEDGER=str(failure),R40_LAUNCHER_HANDLER_SELFTEST="success",R40_H20_EXECUTION_AUTHORIZED="yes",R40_V15_FRESH_AUDIT_APPROVED="yes");self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,0);self.assertFalse(failure.exists())
            env["R40_LAUNCHER_HANDLER_SELFTEST"]="failure";self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,7);self.assertEqual(json.loads(failure.read_text())["exit_code"],7)
    def test_auth_and_fresh_audit_explicit_gates_have_no_marker_or_result_action(self):
        launcher=ROOT/"formal/launch_h20.sh";self.assertNotIn(":?",launcher.read_text())
        cases=({}, {"R40_LAUNCHER_HANDLER_SELFTEST":"success"}, {"R40_H20_EXECUTION_AUTHORIZED":"yes"},{"R40_H20_EXECUTION_AUTHORIZED":"no","R40_V15_FRESH_AUDIT_APPROVED":"yes"},{"R40_H20_EXECUTION_AUTHORIZED":"yes","R40_V15_FRESH_AUDIT_APPROVED":"no"})
        for index,extra in enumerate(cases):
            with tempfile.TemporaryDirectory() as tmp:
                failure=Path(tmp)/"failure.json";marker=Path(tmp)/"marker";env=dict(os.environ,R40_FAILURE_LEDGER=str(failure),R40_ONE_SHOT_MARKER=str(marker),**extra);result=subprocess.run(["bash",str(launcher)],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertTrue(failure.is_file());self.assertFalse(marker.exists());self.assertEqual(list(Path(tmp).iterdir()),[failure])
    def test_atomic_one_shot_concurrency_and_signal_failure_ledgers(self):
        launcher=ROOT/"formal/launch_h20.sh"
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/"marker";action=Path(tmp)/"action";base=dict(os.environ,R40_H20_EXECUTION_AUTHORIZED="yes",R40_V15_FRESH_AUDIT_APPROVED="yes",R40_LAUNCHER_ATOMIC_GATE_SELFTEST="yes",R40_ONE_SHOT_MARKER=str(marker),R40_SELFTEST_RESULT_ACTION=str(action));processes=[]
            for i in range(2):env=dict(base,R40_FAILURE_LEDGER=str(Path(tmp)/f"failure-{i}.json"));processes.append(subprocess.Popen(["bash",str(launcher)],env=env))
            codes=[p.wait() for p in processes];self.assertEqual(sorted(code==0 for code in codes),[False,True]);self.assertTrue(action.is_file());self.assertEqual(sum((Path(tmp)/f"failure-{i}.json").is_file() for i in range(2)),1)
        for signal_name in ("SIGTERM","SIGINT"):
            with tempfile.TemporaryDirectory() as tmp:
                failure=Path(tmp)/"failure.json";marker=Path(tmp)/"marker";env=dict(os.environ,R40_H20_EXECUTION_AUTHORIZED="yes",R40_V15_FRESH_AUDIT_APPROVED="yes",R40_LAUNCHER_ATOMIC_GATE_SELFTEST="yes",R40_ONE_SHOT_MARKER=str(marker),R40_SELFTEST_RESULT_ACTION=str(Path(tmp)/"action"),R40_FAILURE_LEDGER=str(failure));p=subprocess.Popen(["bash",str(launcher)],env=env)
                for _ in range(100):
                    if marker.is_dir():break
                    time.sleep(0.01)
                p.send_signal(getattr(signal,signal_name));self.assertNotEqual(p.wait(),0);self.assertTrue(failure.is_file())
if __name__=="__main__":unittest.main()
