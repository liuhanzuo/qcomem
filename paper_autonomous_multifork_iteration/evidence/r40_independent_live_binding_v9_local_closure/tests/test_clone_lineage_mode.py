from __future__ import annotations
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_real_binding import ActualBindingVerifier,RealBindingError  # noqa:E402
from r40_clone_lineage import CloneLineageMode  # noqa:E402
from test_real_binding import owner  # noqa:E402

SELECTED=json.loads((ROOT/"preregistration.json").read_text())["selected_coordinates"]

class LineageTests(unittest.TestCase):
    def build(self, wrong=False):
        persistent=owner(100); verifier=ActualBindingVerifier(persistent,SELECTED,[0,1,2]); requests=[]
        mode=CloneLineageMode(verifier)
        with mode:
            for request_index in range(8):
                request=owner(0)
                for layer in range(3):
                    for family in ("conv","recurrent"):
                        source_layer=1 if wrong and request_index==0 and layer==2 and family=="conv" else layer
                        getattr(request.layers[layer],family+"_states")[0]=getattr(persistent.layers[source_layer],family+"_states")[0].clone()
                requests.append(request)
        group=SimpleNamespace(requests=tuple(requests),resident_count=8,policy="vllm-q16-shared-document-reuse",audit={"gdn_base_policy":"materialize-request-base-functional-rebind"})
        return verifier,mode,group
    def test_real_torch_dispatch_clone_edges_bind_exact_destinations(self):
        verifier,mode,group=self.build();verifier.verify_built_group(group);receipt=mode.bind_returned_group(group,SELECTED);verifier.attach_lineage_receipt(receipt);self.assertEqual(receipt["selected_bound_edge_count"],5)
    def test_equal_geometry_wrong_source_clone_fails(self):
        verifier,mode,group=self.build(wrong=True)
        with self.assertRaisesRegex(RealBindingError,"coordinate/content|wrong-source"):
            verifier.verify_built_group(group);mode.bind_returned_group(group,SELECTED)
    def test_missing_and_duplicate_edges_fail(self):
        verifier,mode,group=self.build();verifier.verify_built_group(group);mode.edges=[]
        with self.assertRaisesRegex(RealBindingError,"missing or duplicate"):mode.bind_returned_group(group,SELECTED)
        verifier,mode,group=self.build();verifier.verify_built_group(group);mode.edges.append(mode.edges[0])
        with self.assertRaisesRegex(RealBindingError,"missing or duplicate"):mode.bind_returned_group(group,SELECTED)
if __name__=="__main__":unittest.main()
