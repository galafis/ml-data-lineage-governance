"""Tests for src.schema.evolution — SchemaEvolutionTracker."""

import pytest

from src.schema.evolution import (
    ChangeType,
    CompatibilityMode,
    SchemaEvolutionTracker,
)


@pytest.fixture()
def tracker() -> SchemaEvolutionTracker:
    t = SchemaEvolutionTracker("test_ds", CompatibilityMode.BACKWARD)
    t.register_schema(
        {"id": "int64", "name": "object", "value": "float64"},
        "Initial schema",
    )
    return t


class TestChangeDetection:

    def test_detect_added_column(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "name": "object", "value": "float64", "extra": "bool"}
        changes = tracker.detect_changes(tracker.current_schema, new)
        added = [c for c in changes if c.change_type == ChangeType.COLUMN_ADDED]
        assert len(added) == 1
        assert added[0].column == "extra"

    def test_detect_removed_column(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "value": "float64"}
        changes = tracker.detect_changes(tracker.current_schema, new)
        removed = [c for c in changes if c.change_type == ChangeType.COLUMN_REMOVED]
        assert len(removed) == 1
        assert removed[0].column == "name"

    def test_detect_type_change(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "name": "object", "value": "object"}
        changes = tracker.detect_changes(tracker.current_schema, new)
        type_changes = [c for c in changes if c.change_type == ChangeType.TYPE_CHANGED]
        assert len(type_changes) == 1
        assert type_changes[0].column == "value"


class TestCompatibility:

    def test_backward_compatible_add_column(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "name": "object", "value": "float64", "extra": "bool"}
        result = tracker.check_compatibility(new, CompatibilityMode.BACKWARD)
        assert result.compatible is True

    def test_backward_incompatible_remove_column(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "value": "float64"}  # 'name' removed
        result = tracker.check_compatibility(new, CompatibilityMode.BACKWARD)
        assert result.compatible is False

    def test_full_mode_rejects_add_and_remove(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "name": "object", "extra": "bool"}  # value removed, extra added
        result = tracker.check_compatibility(new, CompatibilityMode.FULL)
        assert result.compatible is False


class TestMigration:

    def test_migration_suggestions_generated(self, tracker: SchemaEvolutionTracker):
        new = {"id": "int64", "name": "object", "value": "object", "new_col": "bool"}
        suggestions = tracker.suggest_migrations(new)
        assert any("ADD COLUMN" in s for s in suggestions)
        assert any("ALTER" in s and "TYPE" in s for s in suggestions)

    def test_no_migration_for_identical_schema(self, tracker: SchemaEvolutionTracker):
        same = {"id": "int64", "name": "object", "value": "float64"}
        suggestions = tracker.suggest_migrations(same)
        assert suggestions == ["No migrations needed; schemas are identical."]
