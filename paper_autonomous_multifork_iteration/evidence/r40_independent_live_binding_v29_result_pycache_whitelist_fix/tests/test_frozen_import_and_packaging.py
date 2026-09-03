from __future__ import annotations
import hashlib,importlib,importlib.util,json,os,signal,subprocess,sys,tarfile,tempfile,time,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];REPO=Path(os.environ.get("R40_V29_REPO_ROOT",ROOT.parents[2])).resolve();GPU=REPO/"paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu"
BUILDER_SPEC=importlib.util.spec_from_file_location("r40_v29_launcher_builder_test",ROOT/"scripts/build_formal_launcher.py")
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
LAUNCHER_BUILDER=importlib.util.module_from_spec(BUILDER_SPEC);BUILDER_SPEC.loader.exec_module(LAUNCHER_BUILDER)
LAUNCHER_CONTROL_ENV=frozenset({"R40_FAILURE_LEDGER","R40_H20_EXECUTION_AUTHORIZED","R40_V19_FRESH_AUDIT_APPROVED","R40_V20_FRESH_AUDIT_APPROVED","R40_V21_FRESH_AUDIT_APPROVED","R40_V23_FRESH_AUDIT_APPROVED","R40_V24_FRESH_AUDIT_APPROVED","R40_V25_FRESH_AUDIT_APPROVED","R40_V26_FRESH_AUDIT_APPROVED","R40_V27_FRESH_AUDIT_APPROVED","R40_V28_FRESH_AUDIT_APPROVED","R40_V29_FRESH_AUDIT_APPROVED","R40_ONE_SHOT_MARKER","R40_LAUNCHER_HANDLER_SELFTEST","R40_LAUNCHER_ATOMIC_GATE_SELFTEST","R40_SELFTEST_RESULT_ACTION","R40_SELFTEST_SIGNAL_HOLD"})
def isolated_launcher_env(**updates):
    env={name:value for name,value in os.environ.items() if name not in LAUNCHER_CONTROL_ENV};env.update(updates);return env
def restore_launcher_signal_dispositions():
    for signum in (signal.SIGHUP,signal.SIGINT,signal.SIGQUIT,signal.SIGTERM):signal.signal(signum,signal.SIG_DFL)
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
        spec=importlib.util.spec_from_file_location("r40_v29_cpu_import_regression",runner);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;self.addCleanup(lambda:sys.modules.pop(spec.name,None));spec.loader.exec_module(module)
        self.assertTrue(hasattr(module,"LiveKVIdentityGuard"))
    def test_actual_frozen_builder_import_and_hash(self):
        if not GPU.is_dir():
            self.skipTest("repository-side frozen production dependency is verified at staging/runtime")
        sys.path.insert(0,str(GPU));self.addCleanup(lambda:sys.path.remove(str(GPU)))
        module=importlib.import_module("qcomem_vllm_paged_multifork_resident")
        self.assertTrue(callable(module._prepare_request_gdn_base));self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),"546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e")
    def test_transparent_isolated_proxy_forces_leading_no_bytecode_and_leaves_source_clean(self):
        wrapper=(ROOT/"executed_source/r40_no_bytecode_python").resolve(strict=True)
        proxy=(REPO/"paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/python_proxy_env/bin/python").resolve(strict=True)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp).resolve();source=directory/"source";source.mkdir();(source/"probe.py").write_text("VALUE=1\n")
            log=directory/"argv.log";fake=directory/"fake-python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$R40_ARGV_LOG"\n');fake.chmod(0o755)
            runner=source/"runner.py";rank_wrapper=source/"rank-wrapper.py";capture=directory/"capture"
            common={**os.environ,"R39_PRIMARY_EXPECTED_RUNNER":str(runner),"R39_PRIMARY_RANK_WRAPPER":str(rank_wrapper),"R39_PRIMARY_CODE_ROOT":str(source),"R39_PRIMARY_RUNTIME_ROOT":str(directory),"R39_PRIMARY_BASE_ROOT":str(directory),"R39_PRIMARY_SOURCE_ROOT":str(directory),"R39_PRIMARY_CAPTURE_ROOT":str(capture),"R40_ARGV_LOG":str(log)}
            forced=dict(common,R39_PRIMARY_REAL_PYTHON=str(wrapper),R40_ACTUAL_REAL_PYTHON=str(fake))
            subprocess.run(["bash",str(proxy),"-I","-c","pass"],env=forced,check=True)
            self.assertEqual(log.read_text().splitlines()[:3],["-B","-I","-c"])
            routed_args=["--stage","shard","--rank","3"]
            subprocess.run(["bash",str(proxy),str(runner),*routed_args],env=forced,check=True)
            self.assertEqual(log.read_text().splitlines(),["-B","-I","-B",str(rank_wrapper),"--code-root",str(source),"--runtime-root",str(directory),"--r39-base-root",str(directory),"--primary-source-root",str(directory),"--runner",str(runner),"--capture-root",str(capture),"--",*routed_args])
            integrated=dict(common,R39_PRIMARY_REAL_PYTHON=str(wrapper),R40_ACTUAL_REAL_PYTHON=sys.executable)
            proof=directory/"python-proof.json";code='import json,pathlib,sys;sys.path.insert(0,sys.argv[1]);import probe;pathlib.Path(sys.argv[2]).write_text(json.dumps({"executable":sys.executable,"argv":sys.argv}))'
            subprocess.run(["bash",str(proxy),"-I","-c",code,str(source),str(proof)],env=integrated,check=True)
            observed=json.loads(proof.read_text());self.assertEqual(observed["executable"],sys.executable);self.assertEqual(observed["argv"],["-c",str(source),str(proof)])
            self.assertFalse((source/"__pycache__").exists())
    def test_launcher_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/"launcher.sh";sentinel=b"do-not-overwrite\n";output.write_bytes(sentinel);base=REPO/"paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh";result=subprocess.run([sys.executable,str(ROOT/"scripts/build_formal_launcher.py"),"--v6",str(base),"--output",str(output)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertEqual(output.read_bytes(),sentinel)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp).resolve();source=directory/"v6.sh";attacker=directory/"attacker.sh";output=directory/"launcher.sh";canonical=(REPO/"paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh").read_bytes();source.write_bytes(canonical);attacker.write_bytes(b"attacker\n");real_open=os.open;opened=[]
            def swapping_open(path,flags,*args,**kwargs):
                descriptor=real_open(path,flags,*args,**kwargs);opened.append(Path(path));os.replace(attacker,source);return descriptor
            with mock.patch.object(LAUNCHER_BUILDER.os,"open",side_effect=swapping_open),mock.patch.object(sys,"argv",["build_formal_launcher.py","--v6",str(source),"--output",str(output)]):
                with self.assertRaisesRegex(RuntimeError,"singly linked regular file|changed during stable descriptor snapshot"):
                    LAUNCHER_BUILDER.main()
            self.assertEqual(opened,[source]);self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp).resolve();source=directory/"v6.sh";output=directory/"launcher.sh";source.write_bytes(b"\xff")
            with mock.patch.object(LAUNCHER_BUILDER,"EXPECTED",hashlib.sha256(b"\xff").hexdigest()),mock.patch.object(sys,"argv",["build_formal_launcher.py","--v6",str(source),"--output",str(output)]):
                with self.assertRaises(UnicodeDecodeError):
                    LAUNCHER_BUILDER.main()
            self.assertFalse(output.exists())
    def test_transformed_launcher_uses_only_exclusive_r40_terminal_publications(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/"launcher.sh";base=REPO/"paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh";subprocess.run([sys.executable,str(ROOT/"scripts/build_formal_launcher.py"),"--v6",str(base),"--output",str(output)],check=True,capture_output=True,text=True);text=output.read_text();self.assertIn("r40_tree_closure.py\" expected-paths",text);self.assertIn("R40_EXPECTED_PATH_ARGS+=(--expected-existing-path",text);self.assertIn("\"${R40_EXPECTED_PATH_ARGS[@]}\"",text);self.assertIn("r40_tree_closure.py\" prepare",text);self.assertIn("r40_tree_closure.py\" complete",text);self.assertIn("r40_tree_closure.py\" close",text);self.assertIn("r40_finalize.py\" --terminal-root \"$RESULT_ROOT\"",text);self.assertIn("r40_cuda_smoke.py\" --root \"$RESULT_ROOT\"",text);self.assertNotIn("touch \"$RESULT_ROOT/COMPLETE\"",text);self.assertNotIn("> \"$R40_FORMAL_ROOT/terminal-closure.json\"",text);subprocess.run(["bash","-n",str(output)],check=True)
    def test_cuda_smoke_command_refuses_preexisting_terminal_output_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();formal=root/"r40-formal";formal.mkdir();output=formal/"cuda-smoke.json";sentinel=b"do-not-overwrite\n";output.write_bytes(sentinel);result=subprocess.run([sys.executable,str(ROOT/"executed_source/r40_cuda_smoke.py"),"--root",str(root),"--runner",str(root/"missing-runner.py"),"--output",str(output)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn("overwrite",result.stderr);self.assertEqual(output.read_bytes(),sentinel)
    def test_deterministic_archive_rebuild_is_byte_identical_and_nonoverwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/"artifact-root";root.mkdir();(root/"packages").mkdir();(root/"executed_source").mkdir();(root/"payload.txt").write_text("payload\n");(root/"executed_source/r40_no_bytecode_python").write_text("#!/bin/sh\n");first=root/"packages/first.tar.gz";second=root/"packages/second.tar.gz";command=[sys.executable,str(ROOT/"scripts/build_deterministic_archive.py"),"--root",str(root),"--output"]
            subprocess.run([*command,str(first)],check=True);subprocess.run([*command,str(second)],check=True);self.assertEqual(first.read_bytes(),second.read_bytes());before=first.read_bytes();result=subprocess.run([*command,str(first)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertEqual(first.read_bytes(),before)
            with tarfile.open(first,"r:gz") as archive:self.assertEqual(archive.getmember("artifact-root/executed_source/r40_no_bytecode_python").mode,0o755)
    def test_safe_archive_rejects_symlink_hardlink_and_fifo_without_output(self):
        for kind in ("symlink","hardlink","fifo"):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/"artifact-root";root.mkdir();(root/"packages").mkdir();source=root/"payload.txt";source.write_text("payload\n");bad=root/"bad"
                if kind=="symlink":bad.symlink_to(source)
                elif kind=="hardlink":os.link(source,bad)
                else:os.mkfifo(bad)
                output=root/"packages/candidate.tar.gz";result=subprocess.run([sys.executable,str(ROOT/"scripts/build_deterministic_archive.py"),"--root",str(root),"--output",str(output)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
    def test_archive_dotdot_output_has_no_outside_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/"artifact-root";root.mkdir();(root/"packages").mkdir();(root/"payload.txt").write_text("payload\n");outside=root.parent/"outside.tar.gz";result=subprocess.run([sys.executable,str(ROOT/"scripts/build_deterministic_archive.py"),"--root",str(root),"--output",str(root/".."/outside.name)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn("dotdot",result.stderr);self.assertFalse(outside.exists())
    def test_single_exit_handler_success_and_failure(self):
        launcher=ROOT/"formal/launch_h20.sh";text=launcher.read_text();self.assertEqual(text.count("trap on_exit EXIT"),1);self.assertNotIn("trap 'rm",text)
        with tempfile.TemporaryDirectory() as tmp:
            failure=Path(tmp)/"failure.json";env=isolated_launcher_env(R40_FAILURE_LEDGER=str(failure),R40_LAUNCHER_HANDLER_SELFTEST="success",R40_H20_EXECUTION_AUTHORIZED="yes",R40_V29_FRESH_AUDIT_APPROVED="yes");self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,0);self.assertFalse(failure.exists())
            env["R40_LAUNCHER_HANDLER_SELFTEST"]="failure";self.assertEqual(subprocess.run(["bash",str(launcher)],env=env).returncode,7);self.assertEqual(json.loads(failure.read_text())["exit_code"],7)
    def test_auth_and_fresh_audit_explicit_gates_have_no_marker_or_result_action(self):
        launcher=ROOT/"formal/launch_h20.sh";self.assertNotIn(":?",launcher.read_text())
        cases=({}, {"R40_LAUNCHER_HANDLER_SELFTEST":"success"}, {"R40_H20_EXECUTION_AUTHORIZED":"yes"},{"R40_H20_EXECUTION_AUTHORIZED":"no","R40_V29_FRESH_AUDIT_APPROVED":"yes"},{"R40_H20_EXECUTION_AUTHORIZED":"yes","R40_V29_FRESH_AUDIT_APPROVED":"no"})
        for index,extra in enumerate(cases):
            with tempfile.TemporaryDirectory() as tmp:
                failure=Path(tmp)/"failure.json";marker=Path(tmp)/"marker";env=isolated_launcher_env(R40_FAILURE_LEDGER=str(failure),R40_ONE_SHOT_MARKER=str(marker),**extra);result=subprocess.run(["bash",str(launcher)],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertTrue(failure.is_file());self.assertFalse(marker.exists());self.assertEqual(list(Path(tmp).iterdir()),[failure])
    def test_atomic_one_shot_concurrency_and_signal_failure_ledgers(self):
        launcher=ROOT/"formal/launch_h20.sh"
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/"marker";action=Path(tmp)/"action";base=isolated_launcher_env(R40_H20_EXECUTION_AUTHORIZED="yes",R40_V29_FRESH_AUDIT_APPROVED="yes",R40_LAUNCHER_ATOMIC_GATE_SELFTEST="yes",R40_ONE_SHOT_MARKER=str(marker),R40_SELFTEST_RESULT_ACTION=str(action));processes=[]
            for i in range(2):env=dict(base,R40_FAILURE_LEDGER=str(Path(tmp)/f"failure-{i}.json"));processes.append(subprocess.Popen(["bash",str(launcher)],env=env))
            codes=[p.wait() for p in processes];self.assertEqual(sorted(code==0 for code in codes),[False,True]);self.assertTrue(action.is_file());self.assertEqual(sum((Path(tmp)/f"failure-{i}.json").is_file() for i in range(2)),1)
        for signal_name in ("SIGTERM","SIGINT"):
            with tempfile.TemporaryDirectory() as tmp:
                failure=Path(tmp)/"failure.json";marker=Path(tmp)/"marker";env=isolated_launcher_env(R40_H20_EXECUTION_AUTHORIZED="yes",R40_V29_FRESH_AUDIT_APPROVED="yes",R40_LAUNCHER_ATOMIC_GATE_SELFTEST="yes",R40_SELFTEST_SIGNAL_HOLD="yes",R40_ONE_SHOT_MARKER=str(marker),R40_SELFTEST_RESULT_ACTION=str(Path(tmp)/"action"),R40_FAILURE_LEDGER=str(failure));p=subprocess.Popen(["bash",str(launcher)],env=env,preexec_fn=restore_launcher_signal_dispositions)
                for _ in range(100):
                    if marker.is_dir():break
                    time.sleep(0.01)
                p.send_signal(getattr(signal,signal_name))
                try:code=p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait();self.fail(f"launcher did not honor {signal_name} after explicit child signal reset")
                self.assertNotEqual(code,0);self.assertTrue(failure.is_file())
if __name__=="__main__":unittest.main()
