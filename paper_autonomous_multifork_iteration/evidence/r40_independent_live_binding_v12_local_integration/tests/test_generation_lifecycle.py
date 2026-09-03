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
        with self.assertRaisesRegex(RealBindingError,"missing/duplicate"):v.observe_generation_step(0,1)
        for call_index in range(1,64):
            req=call_index%8;rebind(group,req);v.observe_generation_step(call_index//8,req)
        rebind(group,0)
        with self.assertRaisesRegex(RealBindingError,"schedule/order"):v.observe_generation_step(8,0)

if __name__=="__main__":unittest.main()
