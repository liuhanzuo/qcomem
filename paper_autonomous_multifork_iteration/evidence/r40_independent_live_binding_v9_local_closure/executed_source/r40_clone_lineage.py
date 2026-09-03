from __future__ import annotations

from typing import Any, Mapping
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from r40_real_binding import digest, storage_key, tensor_at, require


class CloneLineageMode(TorchDispatchMode):
    """Capture only clones whose source is an exact frozen persistent tensor."""
    def __init__(self, verifier: Any) -> None:
        super().__init__(); self.verifier=verifier; self.edges=[]
        self.source_by_storage={value["storage_key"]:coord for coord,value in verifier.source.items()}
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}; result=func(*args,**kwargs)
        if str(func)=="aten.clone.default" and args and isinstance(args[0],torch.Tensor) and isinstance(result,torch.Tensor):
            source=args[0]; coord=self.source_by_storage.get(storage_key(source))
            if coord is not None:
                self.edges.append({"coord":coord,"source":source,"destination":result})
        return result

    def bind_returned_group(self, group: Any, selected: list[Mapping[str,Any]]) -> dict[str,Any]:
        rows=[]; used=set()
        for semantic in selected:
            if semantic["owner_kind"] != "request": continue
            dest=tensor_at(group.requests[int(semantic["request_index"])],semantic)
            matches=[(i,e) for i,e in enumerate(self.edges) if storage_key(e["destination"])==storage_key(dest)]
            require(len(matches)==1,"missing or duplicate clone destination edge")
            index,edge=matches[0]; require(index not in used,"duplicate clone edge reuse"); used.add(index)
            require(edge["coord"]==(int(semantic["layer_index"]),str(semantic["state_family"]),int(semantic["state_index"])),"wrong-source clone lineage")
            rows.append({"request_index":int(semantic["request_index"]),"layer_index":int(semantic["layer_index"]),"state_family":str(semantic["state_family"]),"state_index":int(semantic["state_index"]),"source_storage_key":list(storage_key(edge["source"])),"destination_storage_key":list(storage_key(dest)),"source_content_sha256":digest(edge["source"]),"destination_content_sha256":digest(dest),"clone_operator":"aten.clone.default"})
        return {"schema_version":"forkaudit-r40-clone-lineage-v1","passed":True,"source_independent":True,"captured_persistent_clone_count":len(self.edges),"selected_bound_edge_count":len(rows),"edges":rows}

__all__=["CloneLineageMode"]
