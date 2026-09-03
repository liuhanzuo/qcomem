from __future__ import annotations
import ast, inspect, json, os, signal, subprocess, tempfile, time, unittest
from pathlib import Path
from unittest import mock
import authorized_launcher_v9 as launcher
import fixtures_v9 as fx
import freeze_v9, static_audit_v9, v9_guard, v9_runtime

def kwargs(f):
    r=f["runtime"]
    return dict(binding_path=str(f["binding_path"]),archive_path=str(f["archive"]),source_ledger_path=str(f["ledger"]),snapshot_root=str(f["snapshot"]),execution_contract_path=str(r["contract_path"]),runtime_expectation_path=str(r["expectation_path"]),runner_root=str(r["runner"]),runner_manifest_path=str(r["manifest_path"]),terminal_root=str(f["terminal_root"]),attempt=f["attempt"],run_nonce=f["run_nonce"])

def trusted(f):
    return mock.patch.multiple(v9_guard,_compiled_trust_root=mock.DEFAULT,TRUST_ROOT_PUBLIC_KEY_HEX=f["public"].hex())

class V9SecurityTests(unittest.TestCase):
    def run_valid(self,f):
        with mock.patch.object(v9_guard,"_compiled_trust_root",return_value=None), mock.patch.object(v9_guard,"TRUST_ROOT_PUBLIC_KEY_HEX",f["public"].hex()):
            return launcher.run_authorized_campaign(**kwargs(f))

    def test_01_runtime_exposes_no_authority_or_spawn(self):
        for name in ("_AUTHORIZED_PLAN_TOKEN","AuthorizedPlan","Lifecycle","_spawn_worker","spawn_worker","spawn_workers"):
            self.assertFalse(hasattr(v9_runtime,name),name)
        self.assertNotIn("subprocess",vars(v9_runtime))
        sig=inspect.signature(launcher.run_authorized_campaign)
        self.assertFalse(any("spec" in n or "worker" in n for n in sig.parameters))

    def test_02_structural_single_popen(self):
        result=static_audit_v9.audit(Path(__file__).parent)
        self.assertEqual(result["status"],"PASS",result)
        self.assertEqual(len(result["popen_sites"]),1)

    def test_03_valid_eight_real_workers_and_actual_commitments(self):
        with tempfile.TemporaryDirectory() as td:
            f=fx.signed_fixture(Path(td),real_python=True)
            self.assertEqual(self.run_valid(f),0)
            self.assertEqual(len(list(f["terminal_root"].iterdir())),9)
            rows=[json.loads((f["terminal_root"]/(x+".terminal.json")).read_text()) for x in fx.TERM_IDS]
            first=rows[0]; v9_guard.validate_terminal_tree(f["terminal_root"],first["pre_hashes"],first["post_hashes"],"success")
            for row in rows:
                self.assertEqual(row["status"],"success")
                w=row["lifecycle_receipt"]["workers"]
                self.assertTrue(all(x["authority_match"] and x["pgid"]==x["pid"] for x in w))
                self.assertTrue(all(x["actual_argv"][0]==x["actual_executable"]["path"] for x in w))
                self.assertTrue(all(x["actual_cuda_visible_devices"]==fx.UUID0 for x in w))

    def test_04_replay_same_root_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as td:
            f=fx.signed_fixture(Path(td),real_python=True); self.assertEqual(self.run_valid(f),0)
            before={p.name:(p.read_bytes(),p.stat().st_ino) for p in f["terminal_root"].iterdir()}
            with self.assertRaises(v9_guard.Reject): self.run_valid(f)
            after={p.name:(p.read_bytes(),p.stat().st_ino) for p in f["terminal_root"].iterdir()}
            self.assertEqual(before,after)

    def test_05_replay_to_fresh_root_wrong_attempt_nonce_rejected_prespawn(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); f=fx.signed_fixture(base,real_python=False)
            for mutate in ("root","attempt","nonce"):
                fresh=base/("fresh-"+mutate); fresh.mkdir()
                call=kwargs(f); call["terminal_root"]=str(fresh)
                if mutate=="attempt": call["terminal_root"]=str(f["terminal_root"]); call["attempt"]=2
                if mutate=="nonce": call["terminal_root"]=str(f["terminal_root"]); call["run_nonce"]="b"*64
                with mock.patch.object(v9_guard,"_compiled_trust_root",return_value=None), mock.patch.object(v9_guard,"TRUST_ROOT_PUBLIC_KEY_HEX",f["public"].hex()), mock.patch.object(launcher.subprocess,"Popen",side_effect=AssertionError("spawn reached")):
                    with self.assertRaises(v9_guard.Reject): launcher.run_authorized_campaign(**call)
                self.assertEqual(list(fresh.iterdir()),[])

    def test_06_unsigned_missing_and_shell_contract_fail_before_spawn(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); f=fx.signed_fixture(base,real_python=False)
            variants=[]
            missing=kwargs(f); missing["binding_path"]=str(base/"missing"); variants.append(missing)
            bad=kwargs(f); f["binding_path"].write_bytes(b"{}\n"); variants.append(bad)
            for call in variants:
                with mock.patch.object(launcher.subprocess,"Popen",side_effect=AssertionError("spawn reached")):
                    with self.assertRaises((v9_guard.Reject,OSError)): launcher.run_authorized_campaign(**call)
            self.assertEqual(list(f["terminal_root"].iterdir()),[])

    def test_07_signed_semantic_mismatch_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as td:
            f=fx.signed_fixture(Path(td),real_python=False)
            c=f["runtime"]["contract"]; c["workers"][0]["env"]["CUDA_VISIBLE_DEVICES"]="NOT-A-UUID"
            f["runtime"]["contract_path"].write_bytes(v9_guard.canonical_bytes(c)); fx.resign_fixture(f)
            with mock.patch.object(v9_guard,"_compiled_trust_root",return_value=None), mock.patch.object(v9_guard,"TRUST_ROOT_PUBLIC_KEY_HEX",f["public"].hex()), mock.patch.object(launcher.subprocess,"Popen",side_effect=AssertionError("spawn reached")):
                with self.assertRaises(v9_guard.Reject): launcher.run_authorized_campaign(**kwargs(f))
            self.assertEqual(list(f["terminal_root"].iterdir()),[])

    def test_08_signal_inside_popen_return_window_has_no_orphan(self):
        if not hasattr(os,"fork"): self.skipTest("fork required")
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); f=fx.signed_fixture(base,real_python=True,sleeping=True); pidfile=base/"worker.pid"
            child=os.fork()
            if child==0:
                original=launcher.subprocess.Popen
                def racing(*a,**k):
                    p=original(*a,**k)
                    if len(a)>0 and isinstance(a[0],list) and "run.py" in a[0]:
                        pidfile.write_text(str(p.pid)); os.kill(os.getpid(),signal.SIGTERM)
                    return p
                try:
                    with mock.patch.object(v9_guard,"_compiled_trust_root",return_value=None), mock.patch.object(v9_guard,"TRUST_ROOT_PUBLIC_KEY_HEX",f["public"].hex()), mock.patch.object(launcher.subprocess,"Popen",side_effect=racing):
                        code=launcher.run_authorized_campaign(**kwargs(f))
                    os._exit(code)
                except SystemExit as exc: os._exit(int(exc.code))
                except BaseException: os._exit(99)
            _,status=os.waitpid(child,0); self.assertEqual(os.waitstatus_to_exitcode(status),143)
            wp=int(pidfile.read_text()); time.sleep(.1)
            with self.assertRaises(ProcessLookupError): os.kill(wp,0)
            self.assertEqual(len(list(f["terminal_root"].iterdir())),9)
            rows=[json.loads((f["terminal_root"]/(x+".terminal.json")).read_text()) for x in fx.TERM_IDS]
            first=rows[0]; v9_guard.validate_terminal_tree(f["terminal_root"],first["pre_hashes"],first["post_hashes"],"failure")
            self.assertTrue(any(r["lifecycle_receipt"]["workers"][0]["spawned"] for r in rows))
            self.assertTrue(all(r["status"]=="failure" for r in rows))

    def test_09_terminal_rejects_actual_digest_forgery(self):
        with tempfile.TemporaryDirectory() as td:
            f=fx.signed_fixture(Path(td),real_python=True); self.assertEqual(self.run_valid(f),0)
            p=f["terminal_root"]/(fx.TERM_IDS[0]+".terminal.json"); row=json.loads(p.read_text()); row["lifecycle_receipt"]["workers"][0]["actual_env"]["LC_ALL"]="evil"; p.chmod(0o644); p.write_bytes(v9_guard.canonical_bytes(row))
            with self.assertRaises(v9_guard.Reject): v9_guard.validate_terminal_tree(f["terminal_root"],row["pre_hashes"],row["post_hashes"],"success")

    def test_10_protected_parent_preserves_existing_inode(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/"x"; p.write_bytes(b"old"); ino=p.stat().st_ino
            with v9_runtime.ProtectedParent(root) as pub:
                with self.assertRaises(v9_guard.Reject): pub.publish_many({"x":b"new"})
            self.assertEqual((p.read_bytes(),p.stat().st_ino),(b"old",ino))

    def test_11_build_outputs_deterministic(self):
        root=Path(__file__).parent; a,ma=freeze_v9.build_outputs(root); b,mb=freeze_v9.build_outputs(root)
        self.assertEqual(a,b); self.assertEqual(ma,mb); self.assertEqual(ma["status"],"HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING")

if __name__=="__main__": unittest.main(verbosity=2)
