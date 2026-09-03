from __future__ import annotations

"""Preregistered live-fault campaign controls for ForkAudit.

This module deliberately contains no CUDA or tensor mutation code.  A runner
reaches the injection point named by :class:`InjectionStage`, supplies a small
reversible callback that mutates its live objects, and executes the detector
inside :class:`FaultActivationRegistry.activate`.  Keeping activation generic
lets the same campaign logic cover KV ownership, GDN storage, position/mask,
and dispatch faults without adding mutant branches to the production path.

An escaped mutant is a valid (negative) scientific observation.  It is not an
execution crash and must be preserved in the raw campaign output.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from qcomem_vllm_paged_multifork_resident import RuntimeInvariantError


class FaultCampaignConfigurationError(ValueError):
    """Raised when a campaign attempts an invalid activation."""


class CampaignPhase(str, Enum):
    """Mutually exclusive phases of one campaign case."""

    IDLE = "idle"
    CLEAN = "clean"
    MUTANT = "mutant"


class InjectionStage(str, Enum):
    """Stable lifecycle points at which live state may be mutated."""

    AFTER_REQUEST_CONSTRUCTION = "after_request_construction"
    BEFORE_FIRST_APPEND = "before_first_append"
    AFTER_FIRST_APPEND = "after_first_append"
    AFTER_FIRST_GDN_TRANSITION = "after_first_gdn_transition"
    POSITION_VALIDATION = "position_validation"
    ATTENTION_DISPATCH = "attention_dispatch"


class OutcomeClassification(str, Enum):
    """Complete, non-overlapping outcome classes for campaign execution."""

    CLEAN_PASS = "clean_pass"
    CLEAN_FALSE_POSITIVE = "clean_false_positive"
    DETECTED_EXPECTED_GATE = "detected_expected_gate"
    DETECTED_WRONG_GATE = "detected_wrong_gate"
    ESCAPED = "escaped"
    UNEXPECTED_CRASH = "unexpected_crash"


class ExecutionBoundary(str, Enum):
    """The only boundaries at which a campaign exception can originate."""

    CAMPAIGN_SETUP = "campaign_setup"
    INJECTOR_APPLY = "injector_apply"
    DETECTOR_EXERCISE = "detector_exercise"
    INJECTOR_RESTORE = "injector_restore"


@dataclass(frozen=True)
class MutantSpec:
    """One preregistered live fault and the detector it is meant to test."""

    mutant_id: str
    short_name: str
    fault_class: str
    injection_stage: InjectionStage
    expected_gate_id: str
    injection_protocol: str

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["injection_stage"] = self.injection_stage.value
        return row


M1_RESERVATION_ALIAS = "M1"
M2_WRONG_SEQUENCE = "M2"
M3_TAIL_COW_OMISSION = "M3"
M4_GDN_BASE_ALIAS = "M4"
M5_GDN_PEER_ALIAS = "M5"
M6_POSITION_OFF_BY_ONE = "M6"
M7_MASK_VIOLATION = "M7"
M8_CALLABLE_SWAP = "M8"
M9_DENSE_FALLBACK = "M9"


_MUTANT_ROWS = (
    MutantSpec(
        M1_RESERVATION_ALIAS,
        "private-reservation-alias",
        "kv_ownership",
        InjectionStage.AFTER_REQUEST_CONSTRUCTION,
        "KV_RESERVATION_DISJOINT",
        "Alias one request reservation entry to another physical block.",
    ),
    MutantSpec(
        M2_WRONG_SEQUENCE,
        "wrong-request-sequence",
        "kv_identity",
        InjectionStage.ATTENTION_DISPATCH,
        "KV_SEQUENCE_ID",
        "Bind a request ledger to another live request's sequence.",
    ),
    MutantSpec(
        M3_TAIL_COW_OMISSION,
        "partial-tail-cow-omission",
        "kv_ownership",
        InjectionStage.BEFORE_FIRST_APPEND,
        "KV_TAIL_COW",
        "Temporarily replace partial-tail detachment with a no-op.",
    ),
    MutantSpec(
        M4_GDN_BASE_ALIAS,
        "gdn-persistent-base-alias",
        "gdn_ownership",
        InjectionStage.AFTER_FIRST_GDN_TRANSITION,
        "gdn_completed_vs_base_disjoint",
        "Alias a transitioned request GDN state to the persistent base.",
    ),
    MutantSpec(
        M5_GDN_PEER_ALIAS,
        "gdn-request-peer-alias",
        "gdn_ownership",
        InjectionStage.AFTER_FIRST_GDN_TRANSITION,
        "gdn_completed_vs_peers_disjoint",
        "Alias one transitioned request GDN state to a peer request.",
    ),
    MutantSpec(
        M6_POSITION_OFF_BY_ONE,
        "position-off-by-one",
        "semantic_dispatch",
        InjectionStage.POSITION_VALIDATION,
        "POSITION_CANONICAL_VALUES",
        "Shift one live post-RoPE position value by one.",
    ),
    MutantSpec(
        M7_MASK_VIOLATION,
        "materialized-mask-violation",
        "semantic_dispatch",
        InjectionStage.ATTENTION_DISPATCH,
        "MASK_CONTRACT",
        "Supply a materialized mask to the prevalidated no-mask path.",
    ),
    MutantSpec(
        M8_CALLABLE_SWAP,
        "unified-attention-callable-swap",
        "kernel_identity",
        InjectionStage.ATTENTION_DISPATCH,
        "KERNEL_CALLABLE_ID",
        "Replace the frozen unified_attention callable after ledger creation.",
    ),
    MutantSpec(
        M9_DENSE_FALLBACK,
        "dense-or-full-kv-fallback",
        "kernel_identity",
        InjectionStage.ATTENTION_DISPATCH,
        "KV_PAGED_VIEW",
        "Replace a paged K/V view with a dense or fully materialized value.",
    ),
)

MUTANT_SPECS: Mapping[str, MutantSpec] = MappingProxyType(
    {spec.mutant_id: spec for spec in _MUTANT_ROWS}
)
MUTANT_IDS: Tuple[str, ...] = tuple(MUTANT_SPECS)
EXPECTED_GATE_IDS: Mapping[str, str] = MappingProxyType(
    {mutant_id: spec.expected_gate_id for mutant_id, spec in MUTANT_SPECS.items()}
)


def get_mutant_spec(mutant_id: str) -> MutantSpec:
    """Resolve one fixed ID, rejecting unregistered campaign variants."""

    try:
        return MUTANT_SPECS[mutant_id]
    except KeyError as exc:
        raise FaultCampaignConfigurationError(
            "unknown mutant_id {!r}; expected one of {}".format(mutant_id, MUTANT_IDS)
        ) from exc


@runtime_checkable
class ReversibleInjector(Protocol):
    """No-GPU protocol implemented by a context-manager factory."""

    def __call__(
        self, context: MutableMapping[str, Any]
    ) -> ContextManager[Callable[[], bool]]:
        ...


@dataclass(frozen=True)
class AppliedMutation:
    """Undo and post-exit verification functions returned by an injector."""

    undo: Callable[[], None]
    verify_restored: Callable[[], bool]
    target_binding: Optional["TargetMutationBinding"] = None


@dataclass(frozen=True)
class TargetMutationBinding:
    """Pointer-free target evidence captured by the injector itself.

    The pre- and mutated-state digests are fixed while ``apply`` holds the
    actual target.  ``capture_restored_sha256`` is invoked only after the same
    injector has run ``undo``.  The resulting record is therefore inseparable
    from the lifecycle receipt; a producer cannot attach an unrelated target
    witness later during aggregation.
    """

    mutant_id: str
    case_cell_id: str
    capture_id: str
    target_kind: str
    target_field: str
    pre_sha256: str
    mutated_sha256: str
    capture_restored_sha256: Callable[[], str]
    descriptor_schema: str = "pointer-free-canonical-target-descriptor-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("mutant_id", self.mutant_id),
            ("case_cell_id", self.case_cell_id),
            ("capture_id", self.capture_id),
            ("target_kind", self.target_kind),
            ("target_field", self.target_field),
        ):
            if not isinstance(value, str) or not value:
                raise FaultCampaignConfigurationError(
                    f"target mutation binding {label} is missing"
                )
        if self.descriptor_schema != "pointer-free-canonical-target-descriptor-v1":
            raise FaultCampaignConfigurationError(
                "target mutation descriptor schema drift"
            )
        for label, value in (
            ("pre", self.pre_sha256),
            ("mutated", self.mutated_sha256),
        ):
            if not _is_sha256(value):
                raise FaultCampaignConfigurationError(
                    f"target mutation {label} digest is not SHA-256"
                )
        if self.pre_sha256 == self.mutated_sha256:
            raise FaultCampaignConfigurationError("target mutation was a no-op")
        if not callable(self.capture_restored_sha256):
            raise FaultCampaignConfigurationError(
                "target mutation restoration capture is not callable"
            )

    def finalize(self) -> Dict[str, Any]:
        restored = self.capture_restored_sha256()
        if not _is_sha256(restored):
            raise FaultCampaignConfigurationError(
                "restored target digest is not SHA-256"
            )
        if restored != self.pre_sha256:
            raise FaultCampaignConfigurationError(
                "restored target digest differs from pre-mutation state"
            )
        return {
            "schema_version": "qcomem-mutant-target-binding-v2",
            "mutant_id": self.mutant_id,
            "case_cell_id": self.case_cell_id,
            "capture_id": self.capture_id,
            "target_kind": self.target_kind,
            "target_field": self.target_field,
            "descriptor_schema": self.descriptor_schema,
            "expected_change_relation": "pre!=mutated;restored==pre",
            "pre_sha256": self.pre_sha256,
            "mutated_sha256": self.mutated_sha256,
            "restored_sha256": restored,
            "mutation_observed": True,
            "restoration_observed": True,
            "contains_absolute_pointer": False,
        }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_target_mutation_binding(
    value: Any,
    *,
    mutant_id: str,
    case_cell_id: str,
    target_kind: str,
    target_field: str,
) -> Dict[str, Any]:
    """Validate the finalized binding embedded in a MutationReceipt."""

    required = {
        "schema_version",
        "mutant_id",
        "case_cell_id",
        "capture_id",
        "target_kind",
        "target_field",
        "descriptor_schema",
        "expected_change_relation",
        "pre_sha256",
        "mutated_sha256",
        "restored_sha256",
        "mutation_observed",
        "restoration_observed",
        "contains_absolute_pointer",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise FaultCampaignConfigurationError(
            "target mutation binding schema drift"
        )
    if value["schema_version"] != "qcomem-mutant-target-binding-v2":
        raise FaultCampaignConfigurationError(
            "target mutation binding version drift"
        )
    if value["mutant_id"] != mutant_id or value["case_cell_id"] != case_cell_id:
        raise FaultCampaignConfigurationError(
            "target mutation binding case identity drift"
        )
    if (value["target_kind"], value["target_field"]) != (
        target_kind,
        target_field,
    ):
        raise FaultCampaignConfigurationError(
            "target mutation binding target contract drift"
        )
    if not isinstance(value["capture_id"], str) or not value["capture_id"]:
        raise FaultCampaignConfigurationError("target capture ID missing")
    if value["descriptor_schema"] != "pointer-free-canonical-target-descriptor-v1":
        raise FaultCampaignConfigurationError(
            "target mutation descriptor schema drift"
        )
    if value["expected_change_relation"] != "pre!=mutated;restored==pre":
        raise FaultCampaignConfigurationError(
            "target mutation change relation drift"
        )
    before = value["pre_sha256"]
    mutated = value["mutated_sha256"]
    restored = value["restored_sha256"]
    if not all(_is_sha256(item) for item in (before, mutated, restored)):
        raise FaultCampaignConfigurationError(
            "target mutation binding digest drift"
        )
    if before == mutated or restored != before:
        raise FaultCampaignConfigurationError(
            "target mutation binding does not prove change and restoration"
        )
    if not (
        value["mutation_observed"] is True
        and value["restoration_observed"] is True
        and value["contains_absolute_pointer"] is False
    ):
        raise FaultCampaignConfigurationError(
            "target mutation binding flags drift"
        )
    return dict(value)


ApplyCallback = Callable[
    [MutableMapping[str, Any]],
    AppliedMutation,
]


def callback_injector(apply: ApplyCallback) -> ReversibleInjector:
    """Adapt ``apply -> (undo, verify)`` to the injector protocol.

    The callback is intentionally unopinionated about the target.  It can
    mutate a reservation table, request sequence, tail-COW method, GDN state,
    position/mask argument, kernel callable, or K/V representation.  ``undo``
    runs on normal return and on every exception; the registry calls the
    verifier only after the context exits.
    """

    @contextmanager
    def inject(context: MutableMapping[str, Any]) -> Iterator[Callable[[], bool]]:
        applied = apply(context)
        if not isinstance(applied, AppliedMutation):
            raise FaultCampaignConfigurationError(
                "injector apply callback must return AppliedMutation"
            )
        if not callable(applied.undo) or not callable(applied.verify_restored):
            raise FaultCampaignConfigurationError(
                "AppliedMutation undo and verify_restored must be callable"
            )
        undo_completed = False

        def verify_after_exit() -> bool:
            restored = undo_completed and bool(applied.verify_restored())
            target_row = (
                None
                if applied.target_binding is None
                else applied.target_binding.finalize()
            )
            setattr(verify_after_exit, "target_mutation_binding", target_row)
            return restored

        setattr(verify_after_exit, "target_mutation_binding", None)

        try:
            yield verify_after_exit
        finally:
            applied.undo()
            undo_completed = True

    return inject


def mapping_value_injector(
    key: str,
    transform: Callable[[MutableMapping[str, Any], Any], Any],
) -> ReversibleInjector:
    """Temporarily replace one context value and restore exact prior state."""

    def apply(context: MutableMapping[str, Any]) -> AppliedMutation:
        if key not in context:
            raise FaultCampaignConfigurationError(
                "injection context has no key {!r}".format(key)
            )
        original = context[key]
        context[key] = transform(context, original)

        def undo() -> None:
            context[key] = original

        def verify_restored() -> bool:
            return key in context and context[key] is original

        return AppliedMutation(undo, verify_restored)

    return callback_injector(apply)


def attribute_value_injector(
    object_key: str,
    attribute: str,
    transform: Callable[[MutableMapping[str, Any], Any], Any],
) -> ReversibleInjector:
    """Temporarily patch an attribute on an object stored in the context."""

    def apply(context: MutableMapping[str, Any]) -> AppliedMutation:
        if object_key not in context:
            raise FaultCampaignConfigurationError(
                "injection context has no object key {!r}".format(object_key)
            )
        target = context[object_key]
        if not hasattr(target, attribute):
            raise FaultCampaignConfigurationError(
                "target {!r} has no attribute {!r}".format(object_key, attribute)
            )
        original = getattr(target, attribute)
        setattr(target, attribute, transform(context, original))

        def undo() -> None:
            setattr(target, attribute, original)

        def verify_restored() -> bool:
            return hasattr(target, attribute) and getattr(target, attribute) is original

        return AppliedMutation(undo, verify_restored)

    return callback_injector(apply)


@dataclass
class MutationReceipt:
    """Lifecycle receipt proving where injection began and restoration ended.

    ``mutation_applied`` means the injector context's ``__enter__`` returned.
    ``restoration_verified`` means its ``__exit__`` completed without raising;
    reversible injector implementations are contractually required to verify
    their own target restoration before returning from ``__exit__``.
    """

    mutant_id: str
    injection_stage: InjectionStage
    injector_factory_started: bool = False
    injector_factory_completed: bool = False
    injector_enter_started: bool = False
    injector_enter_completed: bool = False
    mutation_applied: bool = False
    injector_exit_started: bool = False
    injector_exit_completed: bool = False
    restoration_verifier_present: bool = False
    restoration_verified: bool = False
    target_mutation_binding: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["injection_stage"] = self.injection_stage.value
        return row


@dataclass(frozen=True)
class CampaignOutcome:
    """Serializable result of exactly one clean or mutant execution."""

    phase: CampaignPhase
    classification: OutcomeClassification
    mutant_id: Optional[str]
    mutant_name: Optional[str]
    injection_stage: Optional[InjectionStage]
    expected_gate_id: Optional[str]
    observed_gate_id: Optional[str]
    boundary_gate_id: Optional[str]
    detector_satisfied: bool
    aggregate_eligible: bool
    scientifically_valid: bool
    mutation_receipt: Optional[MutationReceipt]
    exercise_started: bool
    exercise_completed: bool
    restoration_verified: Optional[bool]
    failure_origin: Optional[ExecutionBoundary] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["phase"] = self.phase.value
        row["classification"] = self.classification.value
        if self.injection_stage is not None:
            row["injection_stage"] = self.injection_stage.value
        if self.mutation_receipt is not None:
            row["mutation_receipt"] = self.mutation_receipt.to_dict()
        if self.failure_origin is not None:
            row["failure_origin"] = self.failure_origin.value
        return row


class FaultActivationRegistry:
    """Fail-closed, single-mutant registry with scoped activation.

    A clean phase cannot load or activate any injector.  A mutant phase may
    load exactly its selected preregistered ID and activate it exactly once at
    the registered stage.  Leaving either context clears the registry and the
    injector's own context manager restores the live state.
    """

    def __init__(self) -> None:
        self._phase = CampaignPhase.IDLE
        self._selected_mutant_id: Optional[str] = None
        self._loaded: Dict[str, ReversibleInjector] = {}
        self._active_mutant_id: Optional[str] = None
        self._activation_count = 0
        self._last_mutation_receipt: Optional[MutationReceipt] = None

    @property
    def phase(self) -> CampaignPhase:
        return self._phase

    @property
    def selected_mutant_id(self) -> Optional[str]:
        return self._selected_mutant_id

    @property
    def active_mutant_id(self) -> Optional[str]:
        return self._active_mutant_id

    @property
    def loaded_mutant_ids(self) -> Tuple[str, ...]:
        return tuple(self._loaded)

    @property
    def activation_count(self) -> int:
        return self._activation_count

    @property
    def last_mutation_receipt(self) -> Optional[MutationReceipt]:
        return self._last_mutation_receipt

    @contextmanager
    def campaign(
        self,
        phase: CampaignPhase,
        *,
        mutant_id: Optional[str] = None,
    ) -> Iterator["FaultActivationRegistry"]:
        if self._phase is not CampaignPhase.IDLE:
            raise FaultCampaignConfigurationError("nested campaign phases are forbidden")
        phase = CampaignPhase(phase)
        if phase is CampaignPhase.CLEAN and mutant_id is not None:
            raise FaultCampaignConfigurationError(
                "clean phase cannot select or load a mutant"
            )
        if phase is CampaignPhase.MUTANT:
            if mutant_id is None:
                raise FaultCampaignConfigurationError(
                    "mutant phase requires one preregistered mutant_id"
                )
            get_mutant_spec(mutant_id)
        elif phase is not CampaignPhase.CLEAN:
            raise FaultCampaignConfigurationError(
                "campaign phase must be clean or mutant"
            )

        self._phase = phase
        self._selected_mutant_id = mutant_id
        self._loaded = {}
        self._active_mutant_id = None
        self._activation_count = 0
        self._last_mutation_receipt = None
        try:
            yield self
        finally:
            self._phase = CampaignPhase.IDLE
            self._selected_mutant_id = None
            self._loaded = {}
            self._active_mutant_id = None
            self._activation_count = 0
            self._last_mutation_receipt = None

    def load(self, mutant_id: str, injector: ReversibleInjector) -> None:
        if self._phase is CampaignPhase.CLEAN:
            raise FaultCampaignConfigurationError(
                "clean phase cannot load a mutant injector"
            )
        if self._phase is not CampaignPhase.MUTANT:
            raise FaultCampaignConfigurationError(
                "mutant injector can be loaded only inside a mutant phase"
            )
        if mutant_id != self._selected_mutant_id:
            raise FaultCampaignConfigurationError(
                "only the selected mutant may be loaded"
            )
        get_mutant_spec(mutant_id)
        if mutant_id in self._loaded:
            raise FaultCampaignConfigurationError("mutant injector loaded twice")
        if not callable(injector):
            raise FaultCampaignConfigurationError("mutant injector is not callable")
        self._loaded[mutant_id] = injector

    @contextmanager
    def activate(
        self,
        mutant_id: str,
        stage: InjectionStage,
        context: MutableMapping[str, Any],
    ) -> Iterator[MutationReceipt]:
        if self._phase is CampaignPhase.CLEAN:
            raise FaultCampaignConfigurationError(
                "clean phase cannot activate a mutant injector"
            )
        if self._phase is not CampaignPhase.MUTANT:
            raise FaultCampaignConfigurationError(
                "mutant activation requires an active mutant phase"
            )
        if mutant_id != self._selected_mutant_id:
            raise FaultCampaignConfigurationError(
                "activation does not match the selected mutant"
            )
        spec = get_mutant_spec(mutant_id)
        stage = InjectionStage(stage)
        if stage is not spec.injection_stage:
            raise FaultCampaignConfigurationError(
                "{} must activate at {}, not {}".format(
                    mutant_id, spec.injection_stage.value, stage.value
                )
            )
        if mutant_id not in self._loaded:
            raise FaultCampaignConfigurationError("mutant injector was not loaded")
        if self._active_mutant_id is not None:
            raise FaultCampaignConfigurationError("nested mutant activation is forbidden")
        if self._activation_count != 0:
            raise FaultCampaignConfigurationError(
                "one campaign case may activate its mutant only once"
            )

        self._active_mutant_id = mutant_id
        self._activation_count += 1
        receipt = MutationReceipt(mutant_id, stage)
        self._last_mutation_receipt = receipt
        try:
            receipt.injector_factory_started = True
            manager = self._loaded[mutant_id](context)
            receipt.injector_factory_completed = True
            receipt.injector_enter_started = True
            with manager as verify_restored:
                receipt.injector_enter_completed = True
                receipt.mutation_applied = True
                try:
                    yield receipt
                finally:
                    receipt.injector_exit_started = True
            receipt.injector_exit_completed = True
            receipt.restoration_verifier_present = callable(verify_restored)
            if not receipt.restoration_verifier_present:
                raise FaultCampaignConfigurationError(
                    "injector did not provide a restoration verifier"
                )
            receipt.restoration_verified = bool(verify_restored())
            receipt.target_mutation_binding = getattr(
                verify_restored, "target_mutation_binding", None
            )
            if not receipt.restoration_verified:
                raise FaultCampaignConfigurationError(
                    "injector restoration verifier failed"
                )
        finally:
            self._active_mutant_id = None


Exercise = Callable[[MutableMapping[str, Any]], Any]


def _exception_fields(exc: Exception) -> Tuple[str, str]:
    return type(exc).__name__, str(exc)


def run_clean_case(
    exercise: Exercise,
    *,
    context: Optional[MutableMapping[str, Any]] = None,
    registry: Optional[FaultActivationRegistry] = None,
) -> CampaignOutcome:
    """Run an unmodified control and classify any detector firing as an FP."""

    state: MutableMapping[str, Any] = {} if context is None else context
    controls = FaultActivationRegistry() if registry is None else registry
    with controls.campaign(CampaignPhase.CLEAN):
        exercise_started = True
        try:
            exercise(state)
        except RuntimeInvariantError as exc:
            error_type, error_message = _exception_fields(exc)
            return CampaignOutcome(
                phase=CampaignPhase.CLEAN,
                classification=OutcomeClassification.CLEAN_FALSE_POSITIVE,
                mutant_id=None,
                mutant_name=None,
                injection_stage=None,
                expected_gate_id=None,
                observed_gate_id=str(exc.gate_id),
                boundary_gate_id=None,
                detector_satisfied=False,
                aggregate_eligible=False,
                scientifically_valid=True,
                mutation_receipt=None,
                exercise_started=exercise_started,
                exercise_completed=False,
                restoration_verified=None,
                failure_origin=ExecutionBoundary.DETECTOR_EXERCISE,
                error_type=error_type,
                error_message=error_message,
            )
        except Exception as exc:
            error_type, error_message = _exception_fields(exc)
            return CampaignOutcome(
                phase=CampaignPhase.CLEAN,
                classification=OutcomeClassification.UNEXPECTED_CRASH,
                mutant_id=None,
                mutant_name=None,
                injection_stage=None,
                expected_gate_id=None,
                observed_gate_id=None,
                boundary_gate_id=None,
                detector_satisfied=False,
                aggregate_eligible=False,
                scientifically_valid=False,
                mutation_receipt=None,
                exercise_started=exercise_started,
                exercise_completed=False,
                restoration_verified=None,
                failure_origin=ExecutionBoundary.DETECTOR_EXERCISE,
                error_type=error_type,
                error_message=error_message,
            )
    return CampaignOutcome(
        phase=CampaignPhase.CLEAN,
        classification=OutcomeClassification.CLEAN_PASS,
        mutant_id=None,
        mutant_name=None,
        injection_stage=None,
        expected_gate_id=None,
        observed_gate_id=None,
        boundary_gate_id=None,
        detector_satisfied=True,
        aggregate_eligible=True,
        scientifically_valid=True,
        mutation_receipt=None,
        exercise_started=True,
        exercise_completed=True,
        restoration_verified=None,
    )


def run_mutant_case(
    mutant_id: str,
    injector: ReversibleInjector,
    exercise: Exercise,
    *,
    context: Optional[MutableMapping[str, Any]] = None,
    registry: Optional[FaultActivationRegistry] = None,
) -> CampaignOutcome:
    """Activate one fixed mutant and classify the detector outcome.

    ``context`` should contain the live objects after the runner has reached
    ``spec.injection_stage``.  The injector remains active only for
    ``exercise`` and must restore the state when the activation context exits.
    """

    spec = get_mutant_spec(mutant_id)
    state: MutableMapping[str, Any] = {} if context is None else context
    controls = FaultActivationRegistry() if registry is None else registry
    observed_gate_id: Optional[str] = None
    boundary_gate_id: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    classification = OutcomeClassification.ESCAPED
    scientifically_valid = True
    mutation_receipt: Optional[MutationReceipt] = None
    exercise_started = False
    exercise_completed = False
    restoration_verified = False
    exercise_error: Optional[Exception] = None
    boundary_error: Optional[Exception] = None
    failure_origin: Optional[ExecutionBoundary] = None

    with controls.campaign(CampaignPhase.MUTANT, mutant_id=mutant_id):
        try:
            controls.load(mutant_id, injector)
            try:
                with controls.activate(
                    mutant_id, spec.injection_stage, state
                ) as active_receipt:
                    mutation_receipt = active_receipt
                    exercise_started = True
                    try:
                        exercise(state)
                    except Exception as exc:
                        # Catch inside the injector context so restoration is a
                        # separate boundary and cannot impersonate a detector.
                        exercise_error = exc
                    else:
                        exercise_completed = True
            except Exception as exc:
                boundary_error = exc
                mutation_receipt = controls.last_mutation_receipt
                if (
                    mutation_receipt is None
                    or not mutation_receipt.injector_enter_completed
                ):
                    failure_origin = ExecutionBoundary.INJECTOR_APPLY
                else:
                    failure_origin = ExecutionBoundary.INJECTOR_RESTORE
            finally:
                if mutation_receipt is None:
                    mutation_receipt = controls.last_mutation_receipt
                restoration_verified = bool(
                    mutation_receipt is not None
                    and mutation_receipt.restoration_verified
                )
        except Exception as exc:
            boundary_error = exc
            failure_origin = ExecutionBoundary.CAMPAIGN_SETUP

    if boundary_error is not None:
        classification = OutcomeClassification.UNEXPECTED_CRASH
        scientifically_valid = False
        error_type, error_message = _exception_fields(boundary_error)
        if isinstance(boundary_error, RuntimeInvariantError):
            boundary_gate_id = str(boundary_error.gate_id)
        if isinstance(exercise_error, RuntimeInvariantError):
            observed_gate_id = str(exercise_error.gate_id)
    elif exercise_error is not None:
        failure_origin = ExecutionBoundary.DETECTOR_EXERCISE
        error_type, error_message = _exception_fields(exercise_error)
        if isinstance(exercise_error, RuntimeInvariantError):
            observed_gate_id = str(exercise_error.gate_id)
            if observed_gate_id == spec.expected_gate_id:
                classification = OutcomeClassification.DETECTED_EXPECTED_GATE
            else:
                classification = OutcomeClassification.DETECTED_WRONG_GATE
        else:
            classification = OutcomeClassification.UNEXPECTED_CRASH
            scientifically_valid = False

    # A detector result is admissible only after the injector has both entered
    # and restored cleanly.  This closes the __enter__/__exit__ spoofing path.
    if (
        classification
        in (
            OutcomeClassification.DETECTED_EXPECTED_GATE,
            OutcomeClassification.DETECTED_WRONG_GATE,
            OutcomeClassification.ESCAPED,
        )
        and not restoration_verified
    ):
        classification = OutcomeClassification.UNEXPECTED_CRASH
        scientifically_valid = False
        failure_origin = ExecutionBoundary.INJECTOR_RESTORE
        error_type = error_type or "FaultCampaignConfigurationError"
        error_message = error_message or "injector restoration was not verified"

    detector_satisfied = (
        classification is OutcomeClassification.DETECTED_EXPECTED_GATE
        and failure_origin is ExecutionBoundary.DETECTOR_EXERCISE
        and boundary_gate_id is None
        and restoration_verified
    )
    return CampaignOutcome(
        phase=CampaignPhase.MUTANT,
        classification=classification,
        mutant_id=spec.mutant_id,
        mutant_name=spec.short_name,
        injection_stage=spec.injection_stage,
        expected_gate_id=spec.expected_gate_id,
        observed_gate_id=observed_gate_id,
        boundary_gate_id=boundary_gate_id,
        detector_satisfied=detector_satisfied,
        aggregate_eligible=detector_satisfied,
        scientifically_valid=scientifically_valid,
        mutation_receipt=mutation_receipt,
        exercise_started=exercise_started,
        exercise_completed=exercise_completed,
        restoration_verified=restoration_verified,
        failure_origin=failure_origin,
        error_type=error_type,
        error_message=error_message,
    )


def _clean_binding_errors(clean: CampaignOutcome) -> list[str]:
    errors = []
    if clean.phase is not CampaignPhase.CLEAN:
        errors.append("phase")
    if any(
        value is not None
        for value in (
            clean.mutant_id,
            clean.mutant_name,
            clean.injection_stage,
            clean.expected_gate_id,
            clean.boundary_gate_id,
            clean.mutation_receipt,
            clean.restoration_verified,
        )
    ):
        errors.append("clean_mutation_fields")
    if clean.classification is OutcomeClassification.CLEAN_PASS:
        if clean.observed_gate_id is not None:
            errors.append("clean_pass_observed_gate")
        if not (
            clean.detector_satisfied
            and clean.aggregate_eligible
            and clean.scientifically_valid
            and clean.exercise_started
            and clean.exercise_completed
            and clean.failure_origin is None
        ):
            errors.append("clean_pass_flags")
    return errors


def _mutant_binding_errors(
    mutant_id: str, outcome: CampaignOutcome
) -> list[str]:
    spec = get_mutant_spec(mutant_id)
    errors = []
    if outcome.mutant_id != mutant_id:
        errors.append("key_mutant_id")
    if outcome.phase is not CampaignPhase.MUTANT:
        errors.append("phase")
    if outcome.mutant_name != spec.short_name:
        errors.append("spec_name")
    if outcome.injection_stage is not spec.injection_stage:
        errors.append("spec_stage")
    if outcome.expected_gate_id != spec.expected_gate_id:
        errors.append("spec_expected_gate")
    receipt = outcome.mutation_receipt
    if receipt is None:
        errors.append("mutation_receipt_missing")
    else:
        if receipt.mutant_id != mutant_id:
            errors.append("receipt_mutant_id")
        if receipt.injection_stage is not spec.injection_stage:
            errors.append("receipt_stage")
        if outcome.restoration_verified != receipt.restoration_verified:
            errors.append("receipt_restoration")

    complete_receipt = bool(
        receipt is not None
        and receipt.injector_factory_started
        and receipt.injector_factory_completed
        and receipt.injector_enter_started
        and receipt.injector_enter_completed
        and receipt.mutation_applied
        and receipt.injector_exit_started
        and receipt.injector_exit_completed
        and receipt.restoration_verifier_present
        and receipt.restoration_verified
    )

    classification = outcome.classification
    if classification is OutcomeClassification.DETECTED_EXPECTED_GATE:
        if not (
            outcome.observed_gate_id == spec.expected_gate_id
            and outcome.boundary_gate_id is None
            and outcome.failure_origin is ExecutionBoundary.DETECTOR_EXERCISE
            and outcome.detector_satisfied
            and outcome.aggregate_eligible
            and outcome.scientifically_valid
            and outcome.exercise_started
            and not outcome.exercise_completed
            and outcome.restoration_verified is True
            and complete_receipt
        ):
            errors.append("expected_detection_binding")
    elif classification is OutcomeClassification.DETECTED_WRONG_GATE:
        if not (
            outcome.observed_gate_id is not None
            and outcome.observed_gate_id != spec.expected_gate_id
            and outcome.boundary_gate_id is None
            and outcome.failure_origin is ExecutionBoundary.DETECTOR_EXERCISE
            and not outcome.detector_satisfied
            and not outcome.aggregate_eligible
            and outcome.scientifically_valid
            and outcome.exercise_started
            and not outcome.exercise_completed
            and outcome.restoration_verified is True
            and complete_receipt
        ):
            errors.append("wrong_gate_binding")
    elif classification is OutcomeClassification.ESCAPED:
        if not (
            outcome.observed_gate_id is None
            and outcome.boundary_gate_id is None
            and outcome.failure_origin is None
            and not outcome.detector_satisfied
            and not outcome.aggregate_eligible
            and outcome.scientifically_valid
            and outcome.exercise_started
            and outcome.exercise_completed
            and outcome.restoration_verified is True
            and complete_receipt
        ):
            errors.append("escape_binding")
    elif classification is OutcomeClassification.UNEXPECTED_CRASH:
        if not (
            not outcome.detector_satisfied
            and not outcome.aggregate_eligible
            and not outcome.scientifically_valid
            and outcome.failure_origin is not None
            and outcome.error_type is not None
        ):
            errors.append("unexpected_crash_binding")
    else:
        errors.append("mutant_classification")
    return errors


def validate_campaign_outcomes(
    clean: CampaignOutcome,
    mutants: Mapping[str, CampaignOutcome],
) -> Dict[str, Any]:
    """Fail-closed campaign aggregate without hiding escapes or wrong gates."""

    missing = [mutant_id for mutant_id in MUTANT_IDS if mutant_id not in mutants]
    extras = sorted(set(mutants) - set(MUTANT_IDS))
    rows = []
    binding_errors: Dict[str, list[str]] = {}
    clean_errors = _clean_binding_errors(clean)
    if clean_errors:
        binding_errors["clean"] = clean_errors
    for mutant_id in MUTANT_IDS:
        outcome = mutants.get(mutant_id)
        if outcome is not None:
            rows.append(outcome.to_dict())
            errors = _mutant_binding_errors(mutant_id, outcome)
            if errors:
                binding_errors[mutant_id] = errors
    passed = (
        clean.classification is OutcomeClassification.CLEAN_PASS
        and not missing
        and not extras
        and not binding_errors
        and all(
            mutants[mutant_id].classification
            is OutcomeClassification.DETECTED_EXPECTED_GATE
            for mutant_id in MUTANT_IDS
            if mutant_id in mutants
        )
    )
    return {
        "passed": passed,
        "clean": clean.to_dict(),
        "mutants": rows,
        "expected_mutant_ids": list(MUTANT_IDS),
        "missing_mutant_ids": missing,
        "unexpected_mutant_ids": extras,
        "binding_errors": binding_errors,
        "escaped_mutant_ids": [
            mutant_id
            for mutant_id in MUTANT_IDS
            if mutant_id in mutants
            and mutants[mutant_id].classification is OutcomeClassification.ESCAPED
        ],
        "wrong_gate_mutant_ids": [
            mutant_id
            for mutant_id in MUTANT_IDS
            if mutant_id in mutants
            and mutants[mutant_id].classification
            is OutcomeClassification.DETECTED_WRONG_GATE
        ],
        "unexpected_crash_mutant_ids": [
            mutant_id
            for mutant_id in MUTANT_IDS
            if mutant_id in mutants
            and mutants[mutant_id].classification
            is OutcomeClassification.UNEXPECTED_CRASH
        ],
    }


__all__ = [
    "AppliedMutation",
    "CampaignOutcome",
    "CampaignPhase",
    "EXPECTED_GATE_IDS",
    "ExecutionBoundary",
    "FaultActivationRegistry",
    "FaultCampaignConfigurationError",
    "InjectionStage",
    "M1_RESERVATION_ALIAS",
    "M2_WRONG_SEQUENCE",
    "M3_TAIL_COW_OMISSION",
    "M4_GDN_BASE_ALIAS",
    "M5_GDN_PEER_ALIAS",
    "M6_POSITION_OFF_BY_ONE",
    "M7_MASK_VIOLATION",
    "M8_CALLABLE_SWAP",
    "M9_DENSE_FALLBACK",
    "MUTANT_IDS",
    "MUTANT_SPECS",
    "MutationReceipt",
    "MutantSpec",
    "OutcomeClassification",
    "ReversibleInjector",
    "RuntimeInvariantError",
    "TargetMutationBinding",
    "attribute_value_injector",
    "callback_injector",
    "get_mutant_spec",
    "mapping_value_injector",
    "run_clean_case",
    "run_mutant_case",
    "validate_campaign_outcomes",
    "validate_target_mutation_binding",
]
