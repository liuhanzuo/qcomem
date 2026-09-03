"""Fixed CUDA observer constructed inside the audited runner boundary."""
from __future__ import annotations
import hashlib
from v4_guard import Reject,need

class FixedCudaCapture:
    def __init__(self, expected_physical_uuid:str, device_index:int):
        import torch
        need(type(device_index) is int and not isinstance(device_index,bool),'device index')
        self.torch=torch; self.device=torch.device('cuda',device_index)
        torch.cuda.set_device(self.device); torch.cuda.synchronize(self.device)
        props=torch.cuda.get_device_properties(self.device)
        actual=str(getattr(props,'uuid',''))
        need(actual==expected_physical_uuid,'torch physical UUID mismatch')
        self.uuid=actual
    def synchronize(self): self.torch.cuda.synchronize(self.device)
    def tensor_digest(self,tensor):
        need(tensor.is_cuda and tensor.device==self.device,'live CUDA tensor')
        self.synchronize(); x=tensor.detach().contiguous().cpu(); return hashlib.sha256(x.numpy().tobytes()).hexdigest()
    def allocator_H0(self):
        t=self.torch; self.synchronize(); current=t.cuda.memory_allocated(self.device)
        t.cuda.reset_peak_memory_stats(self.device); self.synchronize()
        peak=t.cuda.max_memory_allocated(self.device); need(peak==current,'H0 peak=current')
        return {'current':current,'peak':peak,'device_uuid':self.uuid}

def construct_fixed_capture(expected_physical_uuid,device_index):
    """Only audited construction route; accepts no backend/callback/factory."""
    return FixedCudaCapture(expected_physical_uuid,device_index)
