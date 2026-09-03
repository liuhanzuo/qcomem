from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from qcomem_torch import (
    TorchSplitCausalLM,
    greedy_generate_dense,
    greedy_generate_full_prefix,
    greedy_generate_oracle,
    greedy_generate_replay,
)
from run_downstream import atomic_json, load_samples, prompt_parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-GPU exactness smoke for cached dense/hybrid replay"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    adapter = TorchSplitCausalLM(model)
    sample = load_samples(args.data, 1)[0]
    document_ids, query_ids, *_ = prompt_parts(
        tokenizer, sample, args.max_input_tokens
    )
    document_ids = document_ids.cuda()
    query_ids = query_ids.cuda()
    full_ids = torch.cat([document_ids, query_ids])
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])

    started = time.perf_counter()
    oracle = greedy_generate_oracle(
        adapter,
        document_ids,
        query_ids,
        depth=args.depth,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    dense = greedy_generate_dense(
        adapter,
        full_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    replay_state = adapter.write_lower_replay(document_ids, args.depth)
    replay = greedy_generate_replay(
        adapter,
        replay_state,
        query_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    prefix_state = adapter.write_full_prefix(document_ids)
    prefix = greedy_generate_full_prefix(
        adapter,
        prefix_state,
        query_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )

    document_length = document_ids.shape[-1]
    boundaries = [0, document_length // 3, 2 * document_length // 3]
    documents = [
        document_ids[..., start:end]
        for start, end in zip(
            boundaries,
            [*boundaries[1:], document_length],
        )
    ]
    multi_state = adapter.write_lower_replay_documents(documents, args.depth)
    multi = greedy_generate_replay(
        adapter,
        multi_state,
        query_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    q16_state = replay_state.quantize(
        bits=16,
        attention_bits=16,
        linear_bits=16,
        cache_layer_bits=(16,) * len(replay_state.cache.layers),
    )
    q16 = greedy_generate_replay(
        adapter,
        q16_state,
        query_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    torch.cuda.synchronize()

    matches = {
        "cached_dense": dense == oracle,
        "full_prefix": prefix == oracle,
        "cached_replay": replay == oracle,
        "fixed_order_multidoc": multi == oracle,
        "per_layer_q16": q16 == oracle,
    }
    result = {
        "status": "passed" if all(matches.values()) else "failed",
        "matches_oracle": matches,
        "tokens": {
            "oracle": oracle,
            "cached_dense": dense,
            "full_prefix": prefix,
            "cached_replay": replay,
            "fixed_order_multidoc": multi,
            "per_layer_q16": q16,
        },
        "depth": args.depth,
        "document_tokens": int(document_ids.numel()),
        "query_tokens": int(query_ids.numel()),
        "max_new_tokens": args.max_new_tokens,
        "replay_bytes": replay_state.stored_nbytes,
        "q16_replay_bytes": q16_state.stored_nbytes,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit("cached exactness smoke failed")


if __name__ == "__main__":
    main()
