from __future__ import annotations
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_real_binding import ActualBindingVerifier,RealBindingError
from test_real_binding import clean_objects,selected,serialized

def rebind(group,request_index):
    for layer in group.requests[request_index].layers:
        for mapping in (layer.conv_states,layer.recurrent_states):mapping[0]=mapping[0].clone().add_(1)

class GenerationLifecycle(unittest.TestCase):
    def test_rotation_and_permutation_of_preexisting_target_objects_rejected(self):
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);values=[]
        for layer in group.requests[0].layers:
            values.extend([layer.conv_states[0],layer.recurrent_states[0]])
        rotated=values[1:]+values[:1];cursor=0
        for layer in group.requests[0].layers:
            layer.conv_states[0]=rotated[cursor];layer.recurrent_states[0]=rotated[cursor+1];cursor+=2
        with self.assertRaisesRegex(RealBindingError,"globally fresh|descriptor"):v.observe_generation_step(0,0)
    def test_non_target_descriptor_drift_and_target_unauthorized_offset_rejected(self):
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);rebind(group,0);group.requests[7].layers[1].conv_states[0].as_strided_((1,),(1,),1)
        with self.assertRaisesRegex(RealBindingError,"non-target request changed"):v.observe_generation_step(0,0)
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);rebind(group,0);group.requests[0].layers[1].conv_states[0].as_strided_((1,),(1,),1)
        with self.assertRaisesRegex(RealBindingError,"descriptor/offset/interval"):v.observe_generation_step(0,0)
    def test_early_cross_request_pollution_rejected_even_if_later_overwritten(self):
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);rebind(group,0);group.requests[7].layers[1].recurrent_states[0]=group.requests[7].layers[1].recurrent_states[0].clone().add_(1)
        with self.assertRaisesRegex(RealBindingError,"non-target request changed"):v.observe_generation_step(0,0)
    def test_persistent_peer_and_same_request_alias_rejected(self):
        for kind in ("persistent","peer","same"):
            persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);rebind(group,0)
            if kind=="persistent":group.requests[0].layers[0].conv_states[0]=persistent.layers[0].conv_states[0]
            elif kind=="peer":group.requests[0].layers[0].conv_states[0]=group.requests[1].layers[0].conv_states[0]
            else:group.requests[0].layers[0].conv_states[0]=group.requests[0].layers[0].recurrent_states[0]
            with self.assertRaisesRegex(RealBindingError,"rebind|alias"):v.observe_generation_step(0,0)
    def test_valid_eight_round_round_robin_and_counts(self):
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group);v.verify_serialized_phase(serialized(persistent,group,"setup_pre_transition"),"setup_pre_transition",[])
        for round_index in range(8):
            for request_index in range(8):
                rebind(group,request_index);v.observe_generation_step(round_index,request_index)
        receipt=v.verify_serialized_phase(serialized(persistent,group,"post_generation"),"post_generation",list(range(8)))
        self.assertEqual(receipt["generation_calls_verified"],64);self.assertEqual(receipt["request_rebind_counts"],[8]*8)
    def test_missing_duplicate_wrong_order_request_and_extra_fail(self):
        persistent,group=clean_objects();v=ActualBindingVerifier(persistent,selected());v.verify_built_group(group)
        with self.assertRaisesRegex(RealBindingError,"ledger count"):v.verify_serialized_phase(serialized(persistent,group,"post_transition"),"post_transition",[0])
        rebind(group,0)
        with self.assertRaisesRegex(RealBindingError,"schedule/order"):v.observe_generation_step(0,1)
        v.observe_generation_step(0,0)
        with self.assertRaisesRegex(RealBindingError,"missing/duplicate|globally fresh"):v.observe_generation_step(0,1)
        for call_index in range(1,64):
            req=call_index%8;rebind(group,req);v.observe_generation_step(call_index//8,req)
        rebind(group,0)
        with self.assertRaisesRegex(RealBindingError,"schedule/order"):v.observe_generation_step(8,0)

if __name__=="__main__":unittest.main()
