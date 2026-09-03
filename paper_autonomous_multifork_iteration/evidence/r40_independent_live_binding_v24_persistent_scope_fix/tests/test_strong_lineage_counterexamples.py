from __future__ import annotations
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_passive_clone_lineage import LineageViolation,PassiveCloneLineageMode,PersistentSourceRegistry  # noqa:E402

def owners():
    backing=torch.ones(3,dtype=torch.bfloat16);layer0=SimpleNamespace(conv_states={0:backing[0:2]},recurrent_states={0:torch.ones(2,dtype=torch.bfloat16)});layer1=SimpleNamespace(conv_states={0:backing[1:3]},recurrent_states={0:torch.ones(2,dtype=torch.bfloat16)});return SimpleNamespace(layers=[layer0,layer1])
class StrongCounterexamples(unittest.TestCase):
    def test_direct_record_and_postclone_content_tamper_rejected(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with self.assertRaisesRegex(LineageViolation,"outside __torch_dispatch__"):mode._record(object(),"aten.clone.default",persistent.layers[0].conv_states[0],persistent.layers[0].conv_states[0].clone())
        with mode:a=persistent.layers[0].conv_states[0].clone();b=persistent.layers[1].conv_states[0].clone()
        a.add_(2);request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:a}),SimpleNamespace(conv_states={0:b})])
        with self.assertRaisesRegex(LineageViolation,"content differs"):mode.verify_materialized([request],coords)

    def test_capability_single_issue_cross_verifier_and_replay_rejected(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with mode:a=persistent.layers[0].conv_states[0].clone();b=persistent.layers[1].conv_states[0].clone()
        requests=(SimpleNamespace(layers=[SimpleNamespace(conv_states={0:a}),SimpleNamespace(conv_states={0:b})]),)
        class V:
            def __init__(self):self.group=SimpleNamespace(requests=requests);self.count=0
            def _accept_lineage_edges(self,bound):self.count+=len(bound)
        v=V();selected=[{"owner_kind":"request","request_index":0,"layer_index":0,"state_family":"conv","state_index":0}]
        cap=mode.issue_capability(requests,selected,v)
        with self.assertRaisesRegex(LineageViolation,"only one"):mode.issue_capability(requests,selected,v)
        with self.assertRaisesRegex(LineageViolation,"verifier/group"):cap.consume_into(V())
        cap.consume_into(v);self.assertEqual(v.count,1)
        with self.assertRaisesRegex(LineageViolation,"already consumed"):cap.consume_into(v)
    def test_same_storage_offset_wrong_source_rejected(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with mode:wrong=persistent.layers[1].conv_states[0].clone();correct=persistent.layers[1].conv_states[0].clone()
        request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:wrong}),SimpleNamespace(conv_states={0:correct})])
        with self.assertRaisesRegex(LineageViolation,"wrong source"):mode.verify_materialized([request],coords)
    def test_returned_view_object_not_exact_clone_destination_rejected(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with mode:a=persistent.layers[0].conv_states[0].clone();b=persistent.layers[1].conv_states[0].clone()
        request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:a.view_as(a)}),SimpleNamespace(conv_states={0:b})])
        with self.assertRaisesRegex(LineageViolation,"exactly one captured"):mode.verify_materialized([request],coords)
    def test_unused_extra_rooted_edge_rejected(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with mode:a=persistent.layers[0].conv_states[0].clone();b=persistent.layers[1].conv_states[0].clone();_extra=persistent.layers[0].conv_states[0].clone()
        request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:a}),SimpleNamespace(conv_states={0:b})])
        with self.assertRaisesRegex(LineageViolation,"extra edges"):mode.verify_materialized([request],coords)
    def test_public_mutation_impossible_and_private_forgery_breaks_seal(self):
        persistent=owners();coords=[(0,"conv",0),(1,"conv",0)];registry=PersistentSourceRegistry(persistent,coords);mode=PassiveCloneLineageMode(registry)
        with mode:a=persistent.layers[0].conv_states[0].clone();b=persistent.layers[1].conv_states[0].clone()
        with self.assertRaises(AttributeError):mode.events.append(mode.events[0])
        mode._events.append(mode.events[0])
        request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:a}),SimpleNamespace(conv_states={0:b})])
        with self.assertRaisesRegex(LineageViolation,"ledger length/seal"):mode.verify_materialized([request],coords)
if __name__=="__main__":unittest.main()
