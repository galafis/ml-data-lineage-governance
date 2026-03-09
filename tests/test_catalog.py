"""Tests for src.governance.catalog — MetadataCatalog operations."""

import pytest

from src.governance.catalog import AssetStatus, AssetType, MetadataCatalog


@pytest.fixture()
def catalog() -> MetadataCatalog:
    c = MetadataCatalog()
    c.register_dataset("ds1", "Dataset 1", owner="team-a", tags=["raw", "sales"])
    c.register_model("mdl1", "Model 1", owner="team-b", tags=["credit", "ml"])
    c.register_feature("ft1", "Feature 1", owner="team-b", tags=["credit"])
    return c


class TestRegistration:

    def test_register_and_get(self, catalog: MetadataCatalog):
        entry = catalog.get("ds1")
        assert entry is not None
        assert entry.name == "Dataset 1"

    def test_duplicate_registration_raises(self, catalog: MetadataCatalog):
        with pytest.raises(ValueError, match="already registered"):
            catalog.register_dataset("ds1", "Duplicate")


class TestSearch:

    def test_search_by_tag(self, catalog: MetadataCatalog):
        results = catalog.search(tags=["credit"])
        assert len(results) == 2  # mdl1 and ft1

    def test_search_by_owner(self, catalog: MetadataCatalog):
        results = catalog.search(owner="team-a")
        assert len(results) == 1

    def test_search_by_type(self, catalog: MetadataCatalog):
        results = catalog.search(asset_type=AssetType.MODEL)
        assert len(results) == 1
        assert results[0].asset_id == "mdl1"

    def test_search_by_query(self, catalog: MetadataCatalog):
        results = catalog.search(query="Feature")
        assert len(results) == 1


class TestVersioning:

    def test_update_increments_version(self, catalog: MetadataCatalog):
        catalog.update("ds1", description="Updated description")
        entry = catalog.get("ds1")
        assert entry.version == 2

    def test_deprecate_changes_status(self, catalog: MetadataCatalog):
        catalog.deprecate("ds1", reason="Replaced by v2")
        entry = catalog.get("ds1")
        assert entry.status == AssetStatus.DEPRECATED

    def test_version_history_tracked(self, catalog: MetadataCatalog):
        catalog.update("ds1", description="v2 update")
        catalog.update("ds1", description="v3 update")
        history = catalog.get_version_history("ds1")
        assert len(history) == 3  # initial + 2 updates
