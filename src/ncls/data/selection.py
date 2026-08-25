from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contract import SourceState


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CorpusSelection:
    """从冻结 CorpusPlan 中取出的版本化研究子语料。"""

    name: str
    base_corpus: str
    stage: str
    partition: str
    states_per_stratum: int
    strata: tuple[tuple[str, tuple[str, ...]], ...]
    state_ids: tuple[str, ...]
    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name != "layer-stack-p1-v1":
            raise ValueError("the first corpus selection must be layer-stack-p1-v1")
        if (self.base_corpus, self.stage, self.partition) != (
            "layer-stack-v1",
            "P1",
            "target-visible-v1",
        ):
            raise ValueError("layer-stack-p1-v1 identity fields are frozen")
        expected_strata = tuple(
            (difficulty, tags)
            for difficulty in ("W", "G", "S")
            for tags in ((), ("M",))
        )
        if self.states_per_stratum != 5 or self.strata != expected_strata:
            raise ValueError("layer-stack-p1-v1 strata are frozen at W/G/S x none/M, five each")
        if len(self.state_ids) != 30 or len(set(self.state_ids)) != 30:
            raise ValueError("layer-stack-p1-v1 must freeze exactly 30 unique states")
        if any(re.fullmatch(r"[0-9a-f]{64}", state_id) is None for state_id in self.state_ids):
            raise ValueError("corpus selection state IDs must be lowercase SHA-256 values")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.document).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusSelection":
        expected_fields = {
            "schema",
            "name",
            "base_corpus",
            "stage",
            "partition",
            "states_per_stratum",
            "strata",
            "state_ids",
        }
        if set(value) != expected_fields:
            raise ValueError("corpus selection fields do not match v1")
        if value.get("schema") != {"name": "corpus-selection", "version": 1}:
            raise ValueError("unsupported corpus selection schema")
        raw_strata = value["strata"]
        if not isinstance(raw_strata, list):
            raise ValueError("corpus selection strata must be an array")
        strata = tuple(
            (
                str(item["difficulty_class"]),
                tuple(map(str, item["difficulty_tags"])),
            )
            for item in raw_strata
        )
        return cls(
            name=str(value["name"]),
            base_corpus=str(value["base_corpus"]),
            stage=str(value["stage"]),
            partition=str(value["partition"]),
            states_per_stratum=int(value["states_per_stratum"]),
            strata=strata,
            state_ids=tuple(map(str, value["state_ids"])),
            document=dict(value),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CorpusSelection":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("corpus selection root must be an object")
        return cls.from_dict(value)

    def select_states(self, states: Sequence[SourceState]) -> tuple[SourceState, ...]:
        lookup = {state.state_id: state for state in states}
        unknown = sorted(set(self.state_ids) - set(lookup))
        if unknown:
            raise ValueError(f"corpus selection contains unknown state IDs: {unknown[:4]}")
        selected = tuple(lookup[state_id] for state_id in self.state_ids)
        if any(state.split != 0 for state in selected):
            raise ValueError("P1 selection must only use source-train states")
        counts = {
            stratum: sum(
                state.difficulty_class == stratum[0]
                and tuple(state.difficulty_tags) == stratum[1]
                for state in selected
            )
            for stratum in self.strata
        }
        if any(count != self.states_per_stratum for count in counts.values()):
            raise ValueError(f"P1 selection does not satisfy its frozen strata: {counts}")
        return selected
