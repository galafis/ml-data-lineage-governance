"""
Schema evolution tracker with compatibility analysis.

Detects structural changes between schema versions (added, removed,
renamed, type-changed columns), evaluates backward/forward/full
compatibility, and suggests migration strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CompatibilityMode(Enum):
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


class ChangeType(Enum):
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"


# Safe type widening pairs: (old_type, new_type)
SAFE_TYPE_PROMOTIONS: set[tuple[str, str]] = {
    ("int32", "int64"),
    ("int64", "float64"),
    ("int32", "float64"),
    ("float32", "float64"),
    ("int8", "int16"),
    ("int16", "int32"),
    ("int8", "int32"),
    ("int8", "int64"),
    ("int16", "int64"),
    ("bool", "int64"),
    ("bool", "object"),
    ("int64", "object"),
    ("float64", "object"),
}


@dataclass
class SchemaChange:
    """A single detected schema change between two versions."""

    change_type: ChangeType
    column: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    breaking: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "column": self.column,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "breaking": self.breaking,
            "description": self.description,
        }


@dataclass
class CompatibilityResult:
    """Result of a schema compatibility check."""

    compatible: bool
    mode: str
    breaking_changes: list[dict[str, Any]]
    non_breaking_changes: list[dict[str, Any]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mode": self.mode,
            "breaking_changes": self.breaking_changes,
            "non_breaking_changes": self.non_breaking_changes,
            "summary": self.summary,
        }


@dataclass
class SchemaVersion:
    """Snapshot of a schema at a point in time."""

    version: int
    schema: dict[str, str]
    timestamp: str
    description: str = ""
    changes_from_previous: list[dict[str, Any]] = field(default_factory=list)


class SchemaEvolutionTracker:
    """
    Track and analyse schema changes across dataset versions.

    Maintains a chronological history of schema versions, detects
    changes between consecutive versions, evaluates compatibility
    against configurable modes, and suggests migration actions.
    """

    def __init__(
        self,
        dataset_id: str,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD,
    ) -> None:
        self.dataset_id = dataset_id
        self.compatibility_mode = compatibility_mode
        self._versions: list[SchemaVersion] = []

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def register_schema(
        self,
        schema: dict[str, str],
        description: str = "",
    ) -> SchemaVersion:
        """
        Register a new schema version.

        If previous versions exist, automatically detects changes.
        """
        version_num = len(self._versions) + 1
        changes: list[dict[str, Any]] = []

        if self._versions:
            prev_schema = self._versions[-1].schema
            detected = self.detect_changes(prev_schema, schema)
            changes = [c.to_dict() for c in detected]

        sv = SchemaVersion(
            version=version_num,
            schema=dict(schema),
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            changes_from_previous=changes,
        )
        self._versions.append(sv)
        logger.info(
            "Registered schema v%d for '%s' (%d columns, %d changes)",
            version_num,
            self.dataset_id,
            len(schema),
            len(changes),
        )
        return sv

    @property
    def current_schema(self) -> Optional[dict[str, str]]:
        return self._versions[-1].schema if self._versions else None

    @property
    def current_version(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> list[SchemaVersion]:
        return list(self._versions)

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def detect_changes(
        self,
        old_schema: dict[str, str],
        new_schema: dict[str, str],
        rename_threshold: float = 0.8,
    ) -> list[SchemaChange]:
        """
        Compare two schemas and return a list of detected changes.

        Args:
            old_schema: Previous column-name-to-dtype mapping.
            new_schema: New column-name-to-dtype mapping.
            rename_threshold: Similarity threshold for rename detection
                (Jaccard on character bigrams).
        """
        old_cols = set(old_schema.keys())
        new_cols = set(new_schema.keys())

        added = new_cols - old_cols
        removed = old_cols - new_cols

        changes: list[SchemaChange] = []

        # Attempt rename detection between removed and added columns
        renamed_pairs: list[tuple[str, str]] = []
        if removed and added:
            renamed_pairs = self._detect_renames(
                list(removed), list(added), rename_threshold
            )
            matched_old = {p[0] for p in renamed_pairs}
            matched_new = {p[1] for p in renamed_pairs}
            added -= matched_new
            removed -= matched_old

        for old_name, new_name in renamed_pairs:
            old_type = old_schema[old_name]
            new_type = new_schema[new_name]
            changes.append(
                SchemaChange(
                    change_type=ChangeType.COLUMN_RENAMED,
                    column=f"{old_name} -> {new_name}",
                    old_value=old_type,
                    new_value=new_type,
                    breaking=True,
                    description=f"Column '{old_name}' appears renamed to '{new_name}'",
                )
            )

        for col in sorted(added):
            changes.append(
                SchemaChange(
                    change_type=ChangeType.COLUMN_ADDED,
                    column=col,
                    new_value=new_schema[col],
                    breaking=False,
                    description=f"New column '{col}' ({new_schema[col]})",
                )
            )

        for col in sorted(removed):
            changes.append(
                SchemaChange(
                    change_type=ChangeType.COLUMN_REMOVED,
                    column=col,
                    old_value=old_schema[col],
                    breaking=True,
                    description=f"Column '{col}' removed (was {old_schema[col]})",
                )
            )

        # Type changes in common columns
        common = old_cols & new_cols
        for col in sorted(common):
            if old_schema[col] != new_schema[col]:
                safe = (old_schema[col], new_schema[col]) in SAFE_TYPE_PROMOTIONS
                changes.append(
                    SchemaChange(
                        change_type=ChangeType.TYPE_CHANGED,
                        column=col,
                        old_value=old_schema[col],
                        new_value=new_schema[col],
                        breaking=not safe,
                        description=(
                            f"Column '{col}' type changed from "
                            f"{old_schema[col]} to {new_schema[col]}"
                            + (" (safe widening)" if safe else " (potentially breaking)")
                        ),
                    )
                )

        return changes

    # ------------------------------------------------------------------
    # Compatibility checks
    # ------------------------------------------------------------------

    def check_compatibility(
        self,
        new_schema: dict[str, str],
        mode: Optional[CompatibilityMode] = None,
    ) -> CompatibilityResult:
        """
        Check whether *new_schema* is compatible with the current schema.

        Modes:
            BACKWARD: New consumers can read old data.
                      - Adding optional columns is OK.
                      - Removing or renaming columns is breaking.
            FORWARD: Old consumers can read new data.
                     - Removing optional columns is OK.
                     - Adding required columns is breaking.
            FULL: Both backward and forward compatible.
            NONE: All changes accepted.
        """
        mode = mode or self.compatibility_mode
        if not self._versions:
            return CompatibilityResult(
                compatible=True,
                mode=mode.value,
                breaking_changes=[],
                non_breaking_changes=[],
                summary="No previous schema to compare against",
            )

        current = self._versions[-1].schema
        all_changes = self.detect_changes(current, new_schema)

        breaking: list[SchemaChange] = []
        non_breaking: list[SchemaChange] = []

        for change in all_changes:
            if mode == CompatibilityMode.NONE:
                non_breaking.append(change)
            elif mode == CompatibilityMode.BACKWARD:
                if change.change_type in (
                    ChangeType.COLUMN_REMOVED,
                    ChangeType.COLUMN_RENAMED,
                ):
                    breaking.append(change)
                elif change.change_type == ChangeType.TYPE_CHANGED and change.breaking:
                    breaking.append(change)
                else:
                    non_breaking.append(change)
            elif mode == CompatibilityMode.FORWARD:
                if change.change_type == ChangeType.COLUMN_ADDED:
                    breaking.append(change)
                elif change.change_type == ChangeType.TYPE_CHANGED and change.breaking:
                    breaking.append(change)
                else:
                    non_breaking.append(change)
            elif mode == CompatibilityMode.FULL:
                if change.change_type in (
                    ChangeType.COLUMN_ADDED,
                    ChangeType.COLUMN_REMOVED,
                    ChangeType.COLUMN_RENAMED,
                ):
                    breaking.append(change)
                elif change.change_type == ChangeType.TYPE_CHANGED and change.breaking:
                    breaking.append(change)
                else:
                    non_breaking.append(change)

        compatible = len(breaking) == 0
        summary_parts = []
        if breaking:
            summary_parts.append(f"{len(breaking)} breaking change(s)")
        if non_breaking:
            summary_parts.append(f"{len(non_breaking)} non-breaking change(s)")
        summary = "; ".join(summary_parts) if summary_parts else "No changes detected"

        return CompatibilityResult(
            compatible=compatible,
            mode=mode.value,
            breaking_changes=[c.to_dict() for c in breaking],
            non_breaking_changes=[c.to_dict() for c in non_breaking],
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Migration suggestions
    # ------------------------------------------------------------------

    def suggest_migrations(self, new_schema: dict[str, str]) -> list[str]:
        """
        Suggest concrete migration actions to move from the current
        schema to *new_schema*.
        """
        if not self._versions:
            return ["No existing schema. Register the new schema directly."]

        changes = self.detect_changes(self._versions[-1].schema, new_schema)
        suggestions: list[str] = []

        for change in changes:
            if change.change_type == ChangeType.COLUMN_ADDED:
                suggestions.append(
                    f"ALTER TABLE ADD COLUMN \"{change.column}\" "
                    f"{change.new_value} DEFAULT NULL;"
                )
            elif change.change_type == ChangeType.COLUMN_REMOVED:
                suggestions.append(
                    f"-- WARNING: dropping column \"{change.column}\" is destructive.\n"
                    f"ALTER TABLE DROP COLUMN \"{change.column}\";"
                )
            elif change.change_type == ChangeType.COLUMN_RENAMED:
                old_name, new_name = change.column.split(" -> ")
                suggestions.append(
                    f"ALTER TABLE RENAME COLUMN \"{old_name.strip()}\" "
                    f"TO \"{new_name.strip()}\";"
                )
            elif change.change_type == ChangeType.TYPE_CHANGED:
                suggestions.append(
                    f"ALTER TABLE ALTER COLUMN \"{change.column}\" "
                    f"TYPE {change.new_value};  -- was {change.old_value}"
                )

        if not suggestions:
            suggestions.append("No migrations needed; schemas are identical.")

        return suggestions

    # ------------------------------------------------------------------
    # History / Export
    # ------------------------------------------------------------------

    def get_evolution_timeline(self) -> list[dict[str, Any]]:
        """Return the full version history as a list of dicts."""
        timeline = []
        for sv in self._versions:
            timeline.append(
                {
                    "version": sv.version,
                    "timestamp": sv.timestamp,
                    "description": sv.description,
                    "column_count": len(sv.schema),
                    "changes": sv.changes_from_previous,
                }
            )
        return timeline

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "compatibility_mode": self.compatibility_mode.value,
            "current_version": self.current_version,
            "versions": self.get_evolution_timeline(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        text = text.lower()
        return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        ba = cls._bigrams(a)
        bb = cls._bigrams(b)
        inter = len(ba & bb)
        union = len(ba | bb)
        return inter / union if union else 0.0

    @classmethod
    def _detect_renames(
        cls,
        removed: list[str],
        added: list[str],
        threshold: float,
    ) -> list[tuple[str, str]]:
        """Match removed columns to added columns by name similarity."""
        pairs: list[tuple[float, str, str]] = []
        for old in removed:
            for new in added:
                sim = cls._jaccard(old, new)
                if sim >= threshold:
                    pairs.append((sim, old, new))
        pairs.sort(reverse=True)

        used_old: set[str] = set()
        used_new: set[str] = set()
        result: list[tuple[str, str]] = []
        for _sim, old, new in pairs:
            if old not in used_old and new not in used_new:
                result.append((old, new))
                used_old.add(old)
                used_new.add(new)
        return result

    def __repr__(self) -> str:
        return (
            f"SchemaEvolutionTracker(dataset={self.dataset_id!r}, "
            f"versions={self.current_version})"
        )
