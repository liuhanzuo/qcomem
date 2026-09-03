from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))

import r40_compact_rebind_fix as fix
from r40_passive_clone_lineage import PassiveCloneLineageMode, PersistentSourceRegistry


LINEAR = tuple(range(fix.EXPECTED_LINEAR_LAYERS))


def storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())


def assert_compact(case: unittest.TestCase, tensor: torch.Tensor) -> None:
    case.assertTrue(tensor.is_contiguous())
    case.assertEqual(tensor.storage_offset(), 0)
    case.assertEqual(
        tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size()
    )


class Layer:
    def __init__(self, value: float) -> None:
        self.conv_states = {0: torch.full((1, 2, 4), value)}
        self.recurrent_states = {0: torch.full((1, 2, 2), value)}
        self.recurrent_update_calls = 0

    def update_recurrent_state(
        self, recurrent_states: torch.Tensor, state_idx: int = 0, **_kwargs: object
    ) -> torch.Tensor:
        self.recurrent_update_calls += 1
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states


class Request:
    def __init__(self, value: float) -> None:
        self.layers = [Layer(value + index) for index in LINEAR]


class Group:
    def __init__(
        self,
        count: int,
        gdn_policy: str = "borrow-immutable-base-functional-rebind",
    ) -> None:
        self.requests = tuple(Request(float(index)) for index in range(count))
        self.resident_count = count
        self.audit = {"gdn_base_policy": gdn_policy}


def fake_builder() -> object:
    namespace: dict[str, object] = {
        "GDN_BORROW_IMMUTABLE_BASE": "borrow-immutable-base-functional-rebind",
        "GDN_MATERIALIZE_REQUEST_BASE": "materialize-request-base-functional-rebind",
        "_builder_impl": lambda _cache, _plan, **kwargs: Group(
            kwargs["resident_count"], kwargs.get(
                "gdn_base_policy", "borrow-immutable-base-functional-rebind"
            )
        ),
    }

    def original_prepare(
        _persistent: object,
        _request: object,
        _plan: object,
        *,
        policy: str,
    ) -> dict[str, object]:
        return {"policy": policy, "delegated": True}

    namespace["_prepare_request_gdn_base"] = original_prepare
    exec(
        "def build(cache, plan, **kwargs):\n"
        "    return _builder_impl(cache, plan, **kwargs)\n",
        namespace,
    )
    return namespace["build"]


class FakeBackbone(torch.nn.Module):
    """Minimal 32-token functional / one-token in-place GDN state producer."""

    def __init__(self) -> None:
        super().__init__()
        self.multi_conv_rebind_calls = 0
        self.inplace_conv_update_calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Request,
        use_cache: bool,
    ) -> torch.Tensor:
        assert use_cache
        sequence_length = int(input_ids.shape[-1])
        for layer_index in LINEAR:
            layer = past_key_values.layers[layer_index]
            previous_conv = layer.conv_states[0]
            token_value = torch.full(
                (*previous_conv.shape[:-1], 1),
                float(sequence_length + layer_index),
                dtype=previous_conv.dtype,
            )
            full = torch.cat([previous_conv, token_value], dim=-1)
            if sequence_length == 1:
                # Same state semantics as the frozen torch causal-conv fallback:
                # mutate the mapping endpoint rather than calling update_conv_state.
                previous_conv.copy_(full[..., -previous_conv.shape[-1] :])
                self.inplace_conv_update_calls += 1
            else:
                layer.conv_states[0] = full[..., -previous_conv.shape[-1] :].clone()
                self.multi_conv_rebind_calls += 1

            previous_recurrent = layer.recurrent_states[0]
            backing = torch.empty(previous_recurrent.numel() + 5)
            noncompact = backing[5:].view(previous_recurrent.shape)
            noncompact.copy_(previous_recurrent + 1)
            layer.update_recurrent_state(noncompact)
        return input_ids.float()


class FailingBackbone(torch.nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Request,
        use_cache: bool,
    ) -> torch.Tensor:
        assert use_cache and past_key_values is not None
        raise RuntimeError("expected backbone fault")


class CompactRebindTests(unittest.TestCase):
    def setUp(self) -> None:
        fix._reset_counters()

    def test_fresh_compact_clone_handles_offset_oversized_and_noncontiguous(self) -> None:
        backing = torch.arange(40.0)
        offset = backing[7:19].view(3, 4)
        transposed = torch.arange(12.0).view(3, 4).t()
        for source in (offset, transposed):
            with self.subTest(stride=source.stride(), offset=source.storage_offset()):
                before = source.clone()
                fresh = fix.fresh_compact_state(source)
                self.assertIsNot(fresh, source)
                self.assertNotEqual(storage_key(fresh), storage_key(source))
                self.assertTrue(torch.equal(fresh, before))
                self.assertTrue(torch.equal(source, before))
                assert_compact(self, fresh)

    def test_materialized_setup_is_direct_lineage_final_destination(self) -> None:
        persistent_layers = []
        request_layers = []
        for _index in LINEAR:
            conv_backing = torch.arange(32768.0)
            conv = conv_backing.as_strided(
                (1, 8192, 4), (33546240, 1, 8192)
            )
            recurrent_backing = torch.arange(4.0)
            recurrent = recurrent_backing.as_strided((1, 2, 2), (4, 1, 2))
            persistent_layer = SimpleNamespace(
                conv_states={0: conv}, recurrent_states={0: recurrent}
            )
            persistent_layers.append(persistent_layer)
            request_layers.append(
                SimpleNamespace(
                    conv_states={0: conv}, recurrent_states={0: recurrent}
                )
            )
        persistent = SimpleNamespace(layers=persistent_layers)
        request = SimpleNamespace(layers=request_layers)
        plan = SimpleNamespace(linear_layer_indices=LINEAR)
        coordinates = [
            (index, family, 0)
            for index in LINEAR
            for family in ("conv", "recurrent")
        ]
        registry = PersistentSourceRegistry(persistent, coordinates)
        lineage = PassiveCloneLineageMode(registry)
        with lineage:
            result = fix.canonicalize_materialized_request_base(
                persistent, request, plan
            )
        summary = lineage.verify_materialized(
            [request], coordinates, require_direct_clone=True
        )
        self.assertEqual(result["tensor_count"], 60)
        self.assertEqual(summary["captured_lineage_edges"], 60)
        self.assertTrue(summary["all_edges_direct_aten_clone"])
        self.assertEqual(
            request.layers[0].conv_states[0].stride(), (32768, 4, 1)
        )
        for index in LINEAR:
            assert_compact(self, request.layers[index].conv_states[0])
            assert_compact(self, request.layers[index].recurrent_states[0])

    def test_materialization_helper_delegates_borrow_and_restores_identity(self) -> None:
        build = fake_builder()
        namespace = build.__globals__
        original = namespace["_prepare_request_gdn_base"]
        restore = fix.install_compact_materialization_helper(build)
        installed = namespace["_prepare_request_gdn_base"]
        self.assertIsNot(installed, original)
        value = installed(
            object(),
            object(),
            object(),
            policy="borrow-immutable-base-functional-rebind",
        )
        self.assertEqual(
            value,
            {
                "policy": "borrow-immutable-base-functional-rebind",
                "delegated": True,
            },
        )
        restore()
        self.assertIs(namespace["_prepare_request_gdn_base"], original)

    def test_builder_two_step_construction_receipt_distinguishes_final_policy(self) -> None:
        materialized = "materialize-request-base-functional-rebind"
        borrowed = "borrow-immutable-base-functional-rebind"
        build = fake_builder()
        namespace = build.__globals__
        restore = fix.install_compact_materialization_helper(build)
        prepare = namespace["_prepare_request_gdn_base"]
        persistent = Request(100.0)
        plan = SimpleNamespace(linear_layer_indices=LINEAR)

        def request_from_persistent() -> Request:
            request = Request(0.0)
            for layer_index in LINEAR:
                request.layers[layer_index].conv_states[0] = (
                    persistent.layers[layer_index].conv_states[0]
                )
                request.layers[layer_index].recurrent_states[0] = (
                    persistent.layers[layer_index].recurrent_states[0]
                )
            return request

        try:
            borrowed_request = request_from_persistent()
            prepare(persistent, borrowed_request, plan, policy=borrowed)
            fix.mark_compact_rebind_requests(
                SimpleNamespace(
                    requests=(borrowed_request,),
                    resident_count=1,
                    audit={"gdn_base_policy": borrowed},
                ),
                LINEAR,
            )
            first = fix.compact_rebind_receipt()
            self.assertEqual(first["borrowed_setup_calls_delegated"], 1)
            self.assertEqual(first["borrowed_requests_returned"], 1)
            self.assertEqual(first["materialized_setup_calls_canonicalized"], 0)

            materialized_request = request_from_persistent()
            prepare(persistent, materialized_request, plan, policy=borrowed)
            prepare(persistent, materialized_request, plan, policy=materialized)
            fix.mark_compact_rebind_requests(
                SimpleNamespace(
                    requests=(materialized_request,),
                    resident_count=1,
                    audit={"gdn_base_policy": materialized},
                ),
                LINEAR,
            )
            receipt = fix.compact_rebind_receipt()
            self.assertEqual(receipt["requests_wrapped"], 2)
            self.assertEqual(receipt["borrowed_setup_calls_delegated"], 2)
            self.assertEqual(receipt["borrowed_requests_returned"], 1)
            self.assertEqual(receipt["materialized_requests_returned"], 1)
            self.assertEqual(receipt["materialized_setup_calls_canonicalized"], 1)
            self.assertEqual(receipt["materialized_setup_states_canonicalized"], 60)
            self.assertTrue(
                receipt[
                    "all_request_construction_borrow_steps_delegated_exactly_once"
                ]
            )
            self.assertTrue(
                receipt["all_request_final_policies_accounted_exactly_once"]
            )
        finally:
            restore()

    def test_full_formal_mixed_policy_request_cardinalities_by_fault_assignment(self) -> None:
        materialized = "materialize-request-base-functional-rebind"
        borrowed = "borrow-immutable-base-functional-rebind"
        policies = (materialized, borrowed, materialized, borrowed)
        expected_by_fault_groups = {
            5: {"groups": 34, "requests": 498, "borrowed": 238},
            3: {"groups": 32, "requests": 494, "borrowed": 234},
        }
        for fault_groups, expected in expected_by_fault_groups.items():
            with self.subTest(fault_groups=fault_groups):
                fix._reset_counters()
                build = fake_builder()
                namespace = build.__globals__
                restore = fix.install_compact_materialization_helper(build)
                prepare = namespace["_prepare_request_gdn_base"]
                persistent = Request(100.0)
                plan = SimpleNamespace(linear_layer_indices=LINEAR)

                def execute_group(resident_count: int, policy: str) -> None:
                    requests = []
                    for _request_index in range(resident_count):
                        request = Request(0.0)
                        for layer_index in LINEAR:
                            request.layers[layer_index].conv_states[0] = (
                                persistent.layers[layer_index].conv_states[0]
                            )
                            request.layers[layer_index].recurrent_states[0] = (
                                persistent.layers[layer_index].recurrent_states[0]
                            )
                        # The immutable builder always borrows the persistent base
                        # first.  Materialized-final requests then take one second,
                        # direct compact-clone construction step.
                        prepare(persistent, request, plan, policy=borrowed)
                        if policy == materialized:
                            prepare(persistent, request, plan, policy=materialized)
                        requests.append(request)
                    fix.mark_compact_rebind_requests(
                        SimpleNamespace(
                            requests=tuple(requests),
                            resident_count=resident_count,
                            audit={"gdn_base_policy": policy},
                        ),
                        LINEAR,
                    )

                # Exact max-N priming plus four-arm warmup policy geometry.
                execute_group(32, materialized)
                for policy in policies:
                    execute_group(32, policy)
                # Exact 3 resident counts x 4 arms x memory/witness builds.
                for resident_count in (1, 8, 32):
                    for policy in policies:
                        for _cell_role in ("memory", "witness"):
                            execute_group(resident_count, policy)
                # Rank 0 has five N=2 borrowed fault groups; ranks 1--7 have three.
                for _fault_group in range(fault_groups):
                    execute_group(2, borrowed)

                receipt = fix.compact_rebind_receipt()
                self.assertEqual(receipt["groups_wrapped"], expected["groups"])
                self.assertEqual(receipt["requests_wrapped"], expected["requests"])
                self.assertEqual(
                    receipt["borrowed_setup_calls_delegated"], expected["requests"]
                )
                self.assertEqual(
                    receipt["borrowed_requests_returned"], expected["borrowed"]
                )
                self.assertEqual(receipt["materialized_requests_returned"], 260)
                self.assertEqual(
                    receipt["materialized_setup_calls_canonicalized"], 260
                )
                self.assertEqual(
                    receipt["materialized_setup_states_canonicalized"], 15600
                )
                self.assertTrue(
                    receipt[
                        "all_materialized_requests_directly_compact_cloned_60_states"
                    ]
                )
                self.assertTrue(
                    receipt[
                        "all_request_construction_borrow_steps_delegated_exactly_once"
                    ]
                )
                self.assertTrue(
                    receipt["all_request_final_policies_accounted_exactly_once"]
                )
                restore()
    def test_single_token_prebind_preserves_old_endpoint_and_route(self) -> None:
        group = Group(1)
        request = group.requests[0]
        original_functions = tuple(
            request.layers[index].update_recurrent_state.__func__ for index in LINEAR
        )
        fix.mark_compact_rebind_requests(group, LINEAR)
        backbone = FakeBackbone()
        pre = fix.install_backbone_pre_hook(backbone, LINEAR)
        post = fix.install_backbone_post_hook(backbone, LINEAR)
        old_conv = [request.layers[index].conv_states[0] for index in LINEAR]
        old_recurrent = [request.layers[index].recurrent_states[0] for index in LINEAR]
        old_conv_values = [tensor.clone() for tensor in old_conv]
        old_recurrent_values = [tensor.clone() for tensor in old_recurrent]
        try:
            backbone(
                input_ids=torch.ones((1, 1), dtype=torch.long),
                past_key_values=request,
                use_cache=True,
            )
        finally:
            post.remove()
            pre.remove()

        self.assertEqual(backbone.multi_conv_rebind_calls, 0)
        self.assertEqual(backbone.inplace_conv_update_calls, len(LINEAR))
        for index in LINEAR:
            layer = request.layers[index]
            self.assertIs(layer.update_recurrent_state.__func__, original_functions[index])
            self.assertIsNot(layer.conv_states[0], old_conv[index])
            self.assertNotEqual(storage_key(layer.conv_states[0]), storage_key(old_conv[index]))
            self.assertTrue(torch.equal(old_conv[index], old_conv_values[index]))
            self.assertIsNot(layer.recurrent_states[0], old_recurrent[index])
            self.assertNotEqual(
                storage_key(layer.recurrent_states[0]), storage_key(old_recurrent[index])
            )
            self.assertTrue(
                torch.equal(old_recurrent[index], old_recurrent_values[index])
            )
            assert_compact(self, layer.conv_states[0])
            assert_compact(self, layer.recurrent_states[0])

    def test_aborted_cached_call_is_accounted_without_rebind_or_error_replacement(self) -> None:
        group = Group(1)
        request = group.requests[0]
        fix.mark_compact_rebind_requests(group, LINEAR)
        backbone = FailingBackbone()
        pre = fix.install_backbone_pre_hook(backbone, LINEAR)
        post = fix.install_backbone_post_hook(backbone, LINEAR)
        try:
            with self.assertRaisesRegex(RuntimeError, "expected backbone fault"):
                backbone(
                    input_ids=torch.ones((1, 32), dtype=torch.long),
                    past_key_values=request,
                    use_cache=True,
                )
        finally:
            post.remove()
            pre.remove()
        receipt = fix.compact_rebind_receipt()
        self.assertEqual(receipt["multi_token_cached_calls_observed"], 1)
        self.assertEqual(receipt["single_token_cached_calls_observed"], 0)
        self.assertEqual(receipt["cached_calls_aborted_before_postprocess"], 1)
        self.assertEqual(receipt["cached_calls_postprocessed"], 0)
        self.assertEqual(receipt["recurrent_states_post_rebound"], 0)
        self.assertTrue(receipt["all_cached_calls_accounted_exactly_once"])
        self.assertTrue(
            receipt[
                "all_successful_cached_calls_post_rebound_exactly_30_recurrent_states"
            ]
        )

    def test_32_plus_7x1_lifecycle_is_fresh_compact_and_route_counts_hold(self) -> None:
        group = Group(1)
        request = group.requests[0]
        baseline_group = Group(1)
        baseline_request = baseline_group.requests[0]
        baseline_backbone = FakeBackbone()
        original_functions = tuple(
            request.layers[index].update_recurrent_state.__func__ for index in LINEAR
        )
        fix.mark_compact_rebind_requests(group, LINEAR)
        backbone = FakeBackbone()
        pre = fix.install_backbone_pre_hook(backbone, LINEAR)
        post = fix.install_backbone_post_hook(backbone, LINEAR)
        history: list[torch.Tensor] = []
        historical_storage: set[tuple[int, int]] = set()
        try:
            for step, length in enumerate((32, 1, 1, 1, 1, 1, 1, 1)):
                prior = [
                    (
                        request.layers[index].conv_states[0],
                        request.layers[index].conv_states[0].clone(),
                        request.layers[index].recurrent_states[0],
                        request.layers[index].recurrent_states[0].clone(),
                    )
                    for index in LINEAR
                ]
                if step == 0:
                    history.extend(tensor for pair in prior for tensor in (pair[0], pair[2]))
                    historical_storage.update(storage_key(tensor) for tensor in history)
                output = backbone(
                    input_ids=torch.ones((1, length), dtype=torch.long),
                    past_key_values=request,
                    use_cache=True,
                )
                baseline_output = baseline_backbone(
                    input_ids=torch.ones((1, length), dtype=torch.long),
                    past_key_values=baseline_request,
                    use_cache=True,
                )
                self.assertTrue(torch.equal(output, baseline_output))
                for index in LINEAR:
                    layer = request.layers[index]
                    baseline_layer = baseline_request.layers[index]
                    old_conv, old_conv_value, old_rec, old_rec_value = prior[index]
                    self.assertTrue(torch.equal(old_conv, old_conv_value))
                    self.assertTrue(torch.equal(old_rec, old_rec_value))
                    for endpoint in (layer.conv_states[0], layer.recurrent_states[0]):
                        self.assertNotIn(id(endpoint), {id(item) for item in history})
                        self.assertNotIn(storage_key(endpoint), historical_storage)
                        assert_compact(self, endpoint)
                        history.append(endpoint)
                        historical_storage.add(storage_key(endpoint))
                    self.assertIs(
                        layer.update_recurrent_state.__func__, original_functions[index]
                    )
                    self.assertTrue(
                        torch.equal(layer.conv_states[0], baseline_layer.conv_states[0])
                    )
                    self.assertTrue(
                        torch.equal(
                            layer.recurrent_states[0],
                            baseline_layer.recurrent_states[0],
                        )
                    )
        finally:
            post.remove()
            pre.remove()

        self.assertEqual(backbone.multi_conv_rebind_calls, len(LINEAR))
        self.assertEqual(backbone.inplace_conv_update_calls, 7 * len(LINEAR))
        self.assertEqual(
            sum(request.layers[index].recurrent_update_calls for index in LINEAR),
            8 * len(LINEAR),
        )
        receipt = fix.compact_rebind_receipt()
        self.assertEqual(receipt["single_token_cached_calls_observed"], 7)
        self.assertEqual(receipt["single_token_conv_states_pre_rebound"], 7 * 30)
        self.assertEqual(receipt["cached_calls_postprocessed"], 8)
        self.assertEqual(receipt["recurrent_states_post_rebound"], 8 * 30)
        self.assertIs(receipt["all_single_token_calls_rebound_exactly_30"], True)
        self.assertIs(
            receipt[
                "all_successful_cached_calls_post_rebound_exactly_30_recurrent_states"
            ],
            True,
        )

    def test_rank_wide_install_wraps_multiple_groups_and_restores_hooks(self) -> None:
        backbone = FakeBackbone()
        plan = SimpleNamespace(linear_layer_indices=LINEAR)
        runtime = SimpleNamespace(backbone=backbone, plan=plan)
        runner = SimpleNamespace()
        runner._load_formal_model_runtime = lambda: runtime
        runner.build_resident_request_group = fake_builder()
        runner._convert_persistent = lambda cache, _plan, _document, **_kwargs: (
            cache,
            object(),
        )
        original_load = runner._load_formal_model_runtime
        original_build = runner.build_resident_request_group
        original_convert = runner._convert_persistent
        restore, install = fix.install_compact_rebind_fix(runner)
        self.assertEqual(install.scope, "all-rank-science-builds-and-cached-backbone-calls")
        loaded = runner._load_formal_model_runtime()
        first = runner.build_resident_request_group(None, plan, resident_count=1)
        second = runner.build_resident_request_group(None, plan, resident_count=2)
        for group in (first, second):
            for request in group.requests:
                loaded.backbone(
                    input_ids=torch.ones((1, 32), dtype=torch.long),
                    past_key_values=request,
                    use_cache=True,
                )
                loaded.backbone(
                    input_ids=torch.ones((1, 1), dtype=torch.long),
                    past_key_values=request,
                    use_cache=True,
                )
        receipt = fix.compact_rebind_receipt()
        self.assertEqual(receipt["groups_wrapped"], 2)
        self.assertEqual(receipt["requests_wrapped"], 3)
        self.assertEqual(receipt["recurrent_layer_methods_preserved"], 90)
        restore()
        self.assertIs(runner._load_formal_model_runtime, original_load)
        self.assertIs(runner.build_resident_request_group, original_build)
        self.assertIs(runner._convert_persistent, original_convert)

    def test_document_prefill_bypass_is_exactly_scoped_and_unwrapped_calls_fail(self) -> None:
        backbone = FakeBackbone()
        plan = SimpleNamespace(linear_layer_indices=LINEAR)
        runtime = SimpleNamespace(backbone=backbone, plan=plan)
        runner = SimpleNamespace()
        runner._load_formal_model_runtime = lambda: runtime
        runner.build_resident_request_group = fake_builder()

        def convert(
            active_backbone: FakeBackbone,
            _plan: object,
            document: torch.Tensor,
            **_kwargs: object,
        ) -> tuple[Request, object]:
            persistent = Request(100.0)
            active_backbone(
                input_ids=document,
                past_key_values=persistent,
                use_cache=True,
            )
            return persistent, object()

        runner._convert_persistent = convert
        restore, _install = fix.install_compact_rebind_fix(runner)
        loaded = runner._load_formal_model_runtime()
        try:
            persistent, _conversion = runner._convert_persistent(
                loaded.backbone,
                plan,
                torch.ones((1, 32), dtype=torch.long),
                resident_count=1,
            )
            self.assertIsNone(
                getattr(persistent, "_r40_v26_compact_rebind_indices", None)
            )
            receipt = fix.compact_rebind_receipt()
            self.assertEqual(receipt["persistent_build_scopes_entered"], 1)
            self.assertEqual(receipt["persistent_build_scopes_completed"], 1)
            self.assertEqual(receipt["persistent_document_pre_hooks_bypassed"], 1)
            self.assertEqual(receipt["persistent_document_post_hooks_bypassed"], 1)
            self.assertIs(
                receipt["all_persistent_document_builds_scoped_exactly_once"], True
            )
            with self.assertRaisesRegex(
                fix.CompactRebindError, "unwrapped cache outside persistent-build scope"
            ):
                loaded.backbone(
                    input_ids=torch.ones((1, 32), dtype=torch.long),
                    past_key_values=Request(200.0),
                    use_cache=True,
                )
        finally:
            restore()


if __name__ == "__main__":
    unittest.main()
