from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from qcomem_lora import (
    LoRAConfig,
    LoRALinear,
    PG19WindowDataset,
    ReplayQuantConfig,
    adapter_metadata,
    assert_interface_adapter_semantics,
    assert_replay_adapter_semantics,
    bidirectional_topk_kl,
    cache_tensor_records,
    cached_two_stage_autograd_capability_gate,
    detach_cache_tensors,
    estimate_suffix_lora_parameters,
    find_suffix_lora_targets,
    install_suffix_lora,
    lora_disabled,
    lora_gradient_coverage,
    lora_state_dict,
    load_inference_lora_checkpoint,
    quant_student_suffix_hidden,
    functional_cache_capability_gate,
    set_lora_enabled,
    training_semantics_metadata,
)
from qcomem_torch import quantize_residual
from train_qcomem_lora import (
    FROZEN_LONGBENCH_TEST_V2_SHA256,
    audited_cache_query_position_gate,
    assert_resume_compatible,
    optimizer_state_dtypes,
    reject_frozen_test_data,
)


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(value) for value in text.split()]


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(hidden)))


class TinyModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(width) for _ in range(3)])
        self.head = nn.Linear(width, 5, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)


class TinyCachedSuffix(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.num_layers = 3
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.language_model = type("LanguageModel", (), {"norm": nn.Identity()})()
        self.calls: list[tuple[int, int, bool]] = []

    def make_cache(self) -> dict[str, torch.Tensor]:
        return {}

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values=None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        del start, end
        self.calls.append((hidden.shape[1], position_offset, past_key_values is not None))
        output = hidden * self.scale
        if past_key_values is not None and position_offset == 0:
            past_key_values["document_summary"] = output.mean(dim=1, keepdim=True)
        elif past_key_values is not None:
            output = output + past_key_values["document_summary"]
        return output


class TinyNativeLinearCacheLayer:
    def __init__(self) -> None:
        self.conv_states = {}
        self.recurrent_states = {}
        self.is_conv_states_initialized = {0: False}
        self.is_recurrent_states_initialized = {0: False}
        self.has_previous_state = {0: False}
        self.conv_kernel_size = {}
        self.record_past = False

    def lazy_initialization(
        self,
        *,
        conv_states=None,
        recurrent_states=None,
        state_idx=0,
        conv_kernel_size=None,
    ):
        if conv_states is not None:
            width = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            self.conv_kernel_size[state_idx] = width
            self.conv_states[state_idx] = torch.zeros(
                *conv_states.shape[:-1], width, dtype=conv_states.dtype
            )
            self.is_conv_states_initialized[state_idx] = True
        if recurrent_states is not None:
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized[state_idx] = True


class TinyNativeFullCacheLayer:
    def __init__(self) -> None:
        self.keys = None
        self.values = None

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys if self.keys is None else torch.cat([self.keys, keys], dim=1)
        self.values = values if self.values is None else torch.cat([self.values, values], dim=1)
        return self.keys, self.values


class TinyNativeFunctionalSuffix(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.num_layers = 3
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.language_model = type("LanguageModel", (), {"norm": nn.Identity()})()
        self.config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention", "linear_attention"]
        )
        self.calls = []

    def make_cache(self):
        return SimpleNamespace(
            layers=[
                TinyNativeLinearCacheLayer(),
                TinyNativeFullCacheLayer(),
                TinyNativeLinearCacheLayer(),
            ]
        )

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values=None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        del start, end
        self.calls.append((hidden.shape[1], position_offset))
        output = hidden * self.scale
        for index in (0, 2):
            layer = past_key_values.layers[index]
            if layer.has_previous_state[0]:
                output = output + layer.recurrent_states[0]
            mixed = output.transpose(1, 2)
            layer.update_conv_state(mixed, conv_kernel_size=2)
            recurrent = output.mean(dim=1, keepdim=True)
            layer.update_recurrent_state(recurrent)
        past_key_values.layers[1].update(output, output.square())
        return output


class LoRATrainingTest(unittest.TestCase):
    def test_cache_position_gate_accepts_variable_explicit_query_lengths(self) -> None:
        result = audited_cache_query_position_gate(
            [
                {
                    "hard_gate_passed": True,
                    "query_positions_expected": 73,
                    "query_positions_observed": 73,
                },
                {
                    "hard_gate_passed": True,
                    "query_positions_expected": 211,
                    "query_positions_observed": 211,
                },
            ]
        )
        self.assertTrue(result["hard_gate_passed"])
        self.assertEqual(result["query_positions_by_rank"], [73, 211])
        self.assertEqual(result["minimum_query_positions"], 73)
        self.assertEqual(result["maximum_query_positions"], 211)

    def test_cache_position_gate_rejects_mismatch_and_empty_query(self) -> None:
        for rows in (
            [
                {
                    "hard_gate_passed": True,
                    "query_positions_expected": 73,
                    "query_positions_observed": 72,
                }
            ],
            [
                {
                    "hard_gate_passed": True,
                    "query_positions_expected": 0,
                    "query_positions_observed": 0,
                }
            ],
        ):
            self.assertFalse(
                audited_cache_query_position_gate(rows)["hard_gate_passed"]
            )

    def test_interface_adapter_semantics_are_hard_checked(self) -> None:
        metadata = training_semantics_metadata(
            mode="interface",
            depth=7,
            teacher_kind="dense",
            teacher_source="online",
            quant=ReplayQuantConfig(),
            chunk_size=512,
            overlap=0,
        )
        assert_interface_adapter_semantics(
            metadata, depth=7, chunk_size=512, overlap=0
        )
        with self.assertRaises(ValueError):
            assert_interface_adapter_semantics(
                metadata, depth=7, chunk_size=256, overlap=0
            )

    def test_resume_cannot_cross_training_semantics(self) -> None:
        expected = {
            "model": "/model",
            "data_sha256": "pg19",
            "world_size": 8,
            "training": {"steps": 200, "warmup_steps": 20},
            "semantics": {"mode": "quant", "depth": 7},
            "adapter": {"config": {"rank": 32}},
        }
        assert_resume_compatible(expected, expected)
        interface = {**expected, "semantics": {"mode": "interface", "depth": 7}}
        with self.assertRaises(ValueError):
            assert_resume_compatible(interface, expected)

    def test_replay_adapter_semantics_are_hard_checked(self) -> None:
        metadata = training_semantics_metadata(
            mode="quant",
            depth=7,
            teacher_kind="q16_replay",
            teacher_source="online",
            quant=ReplayQuantConfig(
                residual_bits=4,
                attention_bits=4,
                linear_bits=8,
                cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
            ),
            chunk_size=256,
            overlap=32,
        )
        assert_replay_adapter_semantics(
            metadata,
            depth=7,
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
        )
        with self.assertRaises(ValueError):
            assert_replay_adapter_semantics(
                metadata,
                depth=7,
                residual_bits=4,
                attention_bits=4,
                linear_bits=None,
                cache_layer_bits=(8, 8, 4, 4, 8, 8, 8),
            )

    def test_inference_checkpoint_installs_loads_and_starts_disabled(self) -> None:
        source = TinyModel()
        config = LoRAConfig(rank=2, alpha=4.0, target_suffixes=("q_proj", "o_proj"))
        installed = install_suffix_lora(
            source, source.layers, depth=1, config=config
        )
        with torch.no_grad():
            for name, value in source.named_parameters():
                if name.endswith("lora_b"):
                    value.fill_(0.25)
        metadata = {
            "adapter": adapter_metadata(
                source, installed_modules=installed, config=config
            ),
            "semantics": {"depth": 1, "mode": "quant"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "format": "qcomem_suffix_lora_v1",
                    "step": 7,
                    "lora": lora_state_dict(source),
                    "metadata": metadata,
                },
                path,
            )
            target = TinyModel()
            loaded = load_inference_lora_checkpoint(
                target, target.layers, path
            )
        self.assertEqual(loaded["step"], 7)
        source_state = lora_state_dict(source)
        target_state = lora_state_dict(target)
        self.assertEqual(target_state.keys(), source_state.keys())
        for key in source_state:
            self.assertTrue(torch.equal(target_state[key], source_state[key]))
        self.assertTrue(
            all(
                not module.enabled
                for module in target.modules()
                if isinstance(module, LoRALinear)
            )
        )
        set_lora_enabled(target, True)
        self.assertTrue(
            all(
                module.enabled
                for module in target.modules()
                if isinstance(module, LoRALinear)
            )
        )

    def test_frozen_test_v2_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "pg19.jsonl"
            train.write_text('{"text":"independent book text"}\n')
            reject_frozen_test_data(train)
            disguised = Path(directory) / "renamed.jsonl"
            disguised.write_text("placeholder")
            with unittest.mock.patch(
                "train_qcomem_lora.hashlib.sha256"
            ) as mocked_sha:
                mocked_sha.return_value.hexdigest.return_value = (
                    FROZEN_LONGBENCH_TEST_V2_SHA256
                )
                with self.assertRaises(SystemExit):
                    reject_frozen_test_data(disguised)
        with self.assertRaises(SystemExit):
            reject_frozen_test_data(
                Path("/data/qcomem-longbench-test-v2/longbench_test_v2.jsonl")
            )

    def test_longbench_rows_cannot_be_disguised_as_pg19(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.jsonl"
            path.write_text(
                '{"dataset":"qasper","_source_index":68,'
                '"document_ids":[1],"query_ids":[2]}\n'
            )
            with self.assertRaises(ValueError):
                PG19WindowDataset(
                    path,
                    TinyTokenizer(),
                    context_tokens=1,
                    query_tokens=1,
                    stride=2,
                )

    def test_optimizer_dtype_is_observed(self) -> None:
        parameter = nn.Parameter(torch.ones(3, dtype=torch.float32))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        torch.square(parameter).sum().backward()
        optimizer.step()
        dtypes = optimizer_state_dtypes(optimizer)
        self.assertGreater(dtypes.get("torch.float32", 0), 0)

    def test_bidirectional_kl_zero_and_positive(self) -> None:
        teacher = torch.tensor([[[3.0, 1.0, -2.0], [0.0, 2.0, 1.0]]])
        exact, metrics = bidirectional_topk_kl(teacher, teacher)
        self.assertAlmostEqual(float(exact), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["forward_kl"]), 0.0, places=7)
        candidate = -teacher
        shifted, _ = bidirectional_topk_kl(candidate, teacher)
        self.assertGreater(float(shifted), 0.1)

    def test_suffix_only_lora_and_tiny_gradient(self) -> None:
        torch.manual_seed(7)
        model = TinyModel()
        inputs = torch.randn(2, 4, 8)
        baseline = model(inputs).detach()
        candidates = find_suffix_lora_targets(
            model.layers,
            depth=2,
            target_suffixes=("q_proj", "o_proj"),
        )
        estimate = estimate_suffix_lora_parameters(
            model.layers,
            depth=2,
            target_suffixes=("q_proj", "o_proj"),
            rank=2,
        )
        installed = install_suffix_lora(
            model,
            model.layers,
            depth=2,
            config=LoRAConfig(
                rank=2,
                alpha=4,
                target_suffixes=("q_proj", "o_proj"),
            ),
        )
        self.assertEqual(
            installed, ["layers.2.q_proj", "layers.2.o_proj"]
        )
        self.assertEqual(candidates, installed)
        self.assertEqual(estimate, 64)
        self.assertIsInstance(model.layers[2].q_proj, LoRALinear)
        self.assertNotIsInstance(model.layers[1].q_proj, LoRALinear)
        self.assertTrue(torch.equal(model(inputs), baseline))
        self.assertTrue(
            all(
                "lora_" in name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )

        target = torch.randn_like(baseline)
        loss = torch.square(model(inputs) - target).mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        with lora_disabled(model):
            self.assertTrue(torch.equal(model(inputs), baseline))

        metadata = adapter_metadata(
            model,
            installed_modules=installed,
            config=LoRAConfig(
                rank=2,
                alpha=4,
                target_suffixes=("q_proj", "o_proj"),
            ),
        )
        self.assertEqual(metadata["installed_modules"], installed)
        self.assertGreater(metadata["trainable_parameters"], 0)

        coverage = lora_gradient_coverage(model)
        self.assertEqual(coverage["module_count"], 2)
        self.assertTrue(coverage["all_modules_have_finite_grad"])
        self.assertEqual(coverage["finite_module_count"], 2)
        self.assertFalse(coverage["document_cache_contribution_isolated"])

    def test_pg19_and_offline_target_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.jsonl"
            records = [
                {"id": "book", "text": "0 1 2 3 4 5 6 7 8 9"},
                {
                    "id": "offline",
                    "document_ids": [1, 2, 3, 4],
                    "query_ids": [5, 6],
                    "teacher_topk_indices": [[0, 1], [2, 3]],
                    "teacher_topk_logits": [[3.0, 1.0], [4.0, 2.0]],
                },
            ]
            path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
            dataset = PG19WindowDataset(
                path,
                TinyTokenizer(),
                context_tokens=4,
                query_tokens=2,
                stride=4,
            )
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0].document_ids.tolist(), [0, 1, 2, 3])
            self.assertEqual(dataset[0].query_ids.tolist(), [4, 5])
            self.assertEqual(dataset[-1].teacher_topk_indices.shape, (2, 2))

    def test_per_record_window_cap_preserves_book_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "books.jsonl"
            path.write_text(
                '\n'.join(
                    json.dumps({"id": book, "text": "0 1 2 3 4 5 6 7"})
                    for book in ("a", "b", "c")
                )
                + "\n"
            )
            dataset = PG19WindowDataset(
                path,
                TinyTokenizer(),
                context_tokens=2,
                query_tokens=2,
                stride=2,
                limit=3,
                max_windows_per_record=1,
            )
        self.assertEqual([window.source_id for window in dataset.windows], ["a:0", "b:0", "c:0"])

    def test_quant_fake_dequant_and_semantics(self) -> None:
        residual = torch.randn(1, 5, 64, dtype=torch.bfloat16)
        packed = quantize_residual(residual, bits=4, group_size=64)
        restored = packed.dequantize()
        self.assertEqual(restored.shape, residual.shape)
        self.assertFalse(torch.equal(restored, residual))
        semantics = training_semantics_metadata(
            mode="quant",
            depth=7,
            teacher_kind="q16_replay",
            teacher_source="online",
            quant=ReplayQuantConfig(
                residual_bits=4,
                attention_bits=8,
                linear_bits=8,
                cache_layer_bits=(8, 8, 4, 4, 8, 8, 8),
            ),
            chunk_size=256,
            overlap=32,
        )
        self.assertFalse(semantics["is_qlora"])
        self.assertEqual(semantics["store"]["residual_bits"], 4)
        self.assertFalse(semantics["write_path_trainable"])

    def test_cached_two_stage_suffix_preserves_query_trajectory_and_gradients(self) -> None:
        adapter = TinyCachedSuffix()
        document = torch.randn(1, 5, 4)
        query = torch.randn(1, 3, 4)
        hidden = quant_student_suffix_hidden(
            adapter,
            depth=1,
            document_residual=document,
            query_residual=query,
            execution="cached-two-stage",
        )
        self.assertEqual(hidden.shape, query.shape)
        self.assertEqual(adapter.calls, [(5, 0, True), (3, 5, True)])
        hidden.square().mean().backward()
        self.assertIsNotNone(adapter.scale.grad)
        self.assertGreater(float(adapter.scale.grad.abs()), 0.0)

        semantics = training_semantics_metadata(
            mode="quant",
            depth=7,
            teacher_kind="q16_replay",
            teacher_source="online",
            quant=ReplayQuantConfig(),
            chunk_size=512,
            overlap=0,
            student_suffix_execution="cached-two-stage",
        )
        self.assertEqual(
            semantics["student_suffix_execution"],
            "cached_document_prefill_then_full_query_continuation",
        )
        self.assertFalse(
            semantics["training_deployment_suffix_execution_claimed_equivalent"]
        )
        self.assertTrue(
            semantics["training_deployment_cache_boundary_structurally_aligned"]
        )
        self.assertFalse(
            semantics["cached_two_stage_autograd_capability"]["capability_gate_passed"]
        )

    def test_detached_document_cache_only_backpropagates_query_continuation(self) -> None:
        adapter = TinyCachedSuffix()
        document = torch.randn(1, 5, 4)
        query = torch.randn(1, 3, 4)
        hidden = quant_student_suffix_hidden(
            adapter,
            depth=1,
            document_residual=document,
            query_residual=query,
            execution="detached-document-cache",
        )
        hidden.sum().backward()
        self.assertEqual(adapter.calls, [(5, 0, True), (3, 5, True)])
        self.assertAlmostEqual(
            float(adapter.scale.grad),
            float(query.sum()),
            places=5,
        )
        semantics = training_semantics_metadata(
            mode="quant",
            depth=7,
            teacher_kind="q16_replay",
            teacher_source="online",
            quant=ReplayQuantConfig(),
            chunk_size=512,
            overlap=0,
            student_suffix_execution="detached-document-cache",
        )
        self.assertTrue(semantics["document_cache_detached_before_query"])
        self.assertFalse(semantics["document_prefill_parameter_gradients_enabled"])
        self.assertFalse(
            semantics["training_deployment_suffix_execution_claimed_equivalent"]
        )
        self.assertEqual(
            semantics["student_suffix_execution"],
            "cached_document_prefill_detached_then_full_query_continuation",
        )

        audited_adapter = TinyCachedSuffix()
        audited_hidden, audit = quant_student_suffix_hidden(
            audited_adapter,
            depth=1,
            document_residual=document,
            query_residual=query,
            execution="detached-document-cache",
            return_cache_audit=True,
        )
        self.assertEqual(audited_hidden.shape, query.shape)
        self.assertTrue(audit["hard_gate_passed"])
        self.assertEqual(audit["document_cache_tensor_count"], 1)
        self.assertTrue(audit["detached_cache_storage_disjoint"])
        self.assertTrue(audit["detached_cache_all_tensors_grad_free"])
        self.assertTrue(audit["original_cache_versions_unchanged"])
        self.assertEqual(audit["query_positions_observed"], query.shape[1])

    def test_native_functional_cache_preserves_document_graph_and_rebinds(self) -> None:
        adapter = TinyNativeFunctionalSuffix()
        document = torch.randn(1, 5, 4, requires_grad=True)
        query = torch.randn(1, 3, 4, requires_grad=True)
        hidden, audit = quant_student_suffix_hidden(
            adapter,
            depth=1,
            document_residual=document,
            query_residual=query,
            execution="native-functional-cache",
            return_cache_audit=True,
        )
        hidden.square().mean().backward()
        self.assertEqual(adapter.calls, [(5, 0), (3, 5)])
        self.assertGreater(float(document.grad.abs().max()), 0)
        self.assertGreater(float(query.grad.abs().max()), 0)
        self.assertGreater(float(adapter.scale.grad.abs()), 0)
        self.assertTrue(audit["hard_gate_passed"])
        self.assertTrue(audit["original_cache_versions_unchanged"])
        self.assertTrue(audit["all_cache_paths_rebound"])
        self.assertEqual(audit["native_linear_layer_count"], 2)
        semantics = training_semantics_metadata(
            mode="quant",
            depth=7,
            teacher_kind="q16_replay",
            teacher_source="online",
            quant=ReplayQuantConfig(),
            chunk_size=512,
            overlap=0,
            student_suffix_execution="native-functional-cache",
        )
        self.assertTrue(
            semantics["native_model_kernels_with_functional_cache_writes"]
        )

    def test_detach_cache_tensors_preserves_aliases_and_removes_graph(self) -> None:
        tensor = torch.randn(2, requires_grad=True) * 2
        cache = {"layers": [{"state": tensor}], "alias": tensor}
        detached = detach_cache_tensors(cache)
        self.assertIs(detached["layers"][0]["state"], detached["alias"])
        self.assertFalse(detached["alias"].requires_grad)
        self.assertIsNone(detached["alias"].grad_fn)
        self.assertNotEqual(detached["alias"].data_ptr(), tensor.data_ptr())
        records = cache_tensor_records(detached)
        self.assertEqual(len(records), 1)

    def test_mutable_and_functional_cache_capability_gates_fail_closed(self) -> None:
        mutable = cached_two_stage_autograd_capability_gate()
        functional = functional_cache_capability_gate()
        self.assertFalse(mutable["capability_gate_passed"])
        self.assertEqual(mutable["evidence_trial"], 1830867)
        self.assertTrue(functional["implemented"])
        self.assertFalse(functional["capability_gate_passed"])
        self.assertTrue(functional["tiny_reference"]["implemented"])
        self.assertTrue(
            functional["tiny_reference"][
                "input_and_parameter_gradient_parity_passed"
            ]
        )
        self.assertTrue(functional["qwen35_integration_implemented"])
        self.assertFalse(functional["qwen35_real_model_gate_passed"])


if __name__ == "__main__":
    unittest.main()
