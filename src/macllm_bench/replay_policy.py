from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QuantizationOption:
    """One measured storage/quality point for a replay component."""

    bits: int
    nbytes: int
    distortion: float

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be positive")
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if self.distortion < 0:
            raise ValueError("distortion must be non-negative")


@dataclass(frozen=True)
class ComponentProfile:
    """Measured alternatives for one residual or cache layer."""

    name: str
    options: tuple[QuantizationOption, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("component names must be non-empty")
        if not self.options:
            raise ValueError(f"component {self.name!r} has no options")
        bits = [option.bits for option in self.options]
        if len(bits) != len(set(bits)):
            raise ValueError(f"component {self.name!r} repeats a bit width")


@dataclass(frozen=True)
class ReplayBitPolicy:
    """Budget-feasible bit assignment selected from measured alternatives."""

    choices: tuple[tuple[str, int], ...]
    total_nbytes: int
    total_distortion: float
    budget_bytes: int

    def as_dict(self) -> dict[str, int]:
        return dict(self.choices)


def _pareto_prune(
    states: Iterable[tuple[int, float, tuple[tuple[str, int], ...]]],
) -> list[tuple[int, float, tuple[tuple[str, int], ...]]]:
    """Keep only states not dominated in both bytes and distortion."""

    ordered = sorted(states, key=lambda state: (state[0], state[1], state[2]))
    frontier = []
    best_distortion = float("inf")
    for state in ordered:
        if state[1] < best_distortion:
            frontier.append(state)
            best_distortion = state[1]
    return frontier


def optimize_bit_policy(
    profiles: Iterable[ComponentProfile],
    *,
    budget_bytes: int,
) -> ReplayBitPolicy:
    """Solve the multiple-choice storage budget problem exactly.

    Distortion must be additive across components.  It may be reconstruction
    error, a logit-distance calibration score, or an estimated task loss.
    Pareto pruning keeps the dynamic program compact without quantizing the
    byte budget into coarse buckets.
    """

    profiles = tuple(profiles)
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("component names must be unique")

    states: list[tuple[int, float, tuple[tuple[str, int], ...]]] = [
        (0, 0.0, ())
    ]
    for profile in profiles:
        candidates = []
        for used_bytes, distortion, choices in states:
            for option in profile.options:
                new_bytes = used_bytes + option.nbytes
                if new_bytes <= budget_bytes:
                    candidates.append(
                        (
                            new_bytes,
                            distortion + option.distortion,
                            (*choices, (profile.name, option.bits)),
                        )
                    )
        if not candidates:
            minimum = sum(
                min(option.nbytes for option in item.options)
                for item in profiles
            )
            raise ValueError(
                f"budget {budget_bytes} cannot fit the minimum {minimum} bytes"
            )
        states = _pareto_prune(candidates)

    used_bytes, distortion, choices = min(
        states, key=lambda state: (state[1], state[0], state[2])
    )
    return ReplayBitPolicy(
        choices=choices,
        total_nbytes=used_bytes,
        total_distortion=distortion,
        budget_bytes=budget_bytes,
    )
