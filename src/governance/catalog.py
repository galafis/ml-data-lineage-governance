"""
Metadata catalog for datasets, models, and features.

Provides a searchable, versioned registry of data assets with
tagging, ownership tracking, and version history for governance
and discoverability.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AssetType(Enum):
    DATASET = "dataset"
    MODEL = "model"
    FEATURE = "feature"
    PIPELINE = "pipeline"
    REPORT = "report"
    ARTIFACT = "artifact"


class AssetStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DRAFT = "draft"


class CatalogEntry:
    """
    A single catalogued data asset.

    Tracks the asset's metadata, version history, ownership,
    and arbitrary tags for search/filtering.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        asset_type: AssetType,
        description: str = "",
        owner: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        schema: Optional[dict[str, str]] = None,
    ) -> None:
        self.asset_id = asset_id
        self.name = name
        self.asset_type = asset_type
        self.description = description
        self.owner = owner
        self.tags: list[str] = tags or []
        self.metadata: dict[str, Any] = metadata or {}
        self.schema: Optional[dict[str, str]] = schema
        self.status = AssetStatus.ACTIVE
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.version = 1
        self._version_history: list[dict[str, Any]] = []
        self._record_version("Initial registration")

    def update(
        self,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        schema: Optional[dict[str, str]] = None,
        status: Optional[AssetStatus] = None,
        change_note: str = "",
    ) -> None:
        """Apply an incremental update and bump the version."""
        if description is not None:
            self.description = description
        if owner is not None:
            self.owner = owner
        if tags is not None:
            self.tags = tags
        if metadata is not None:
            self.metadata.update(metadata)
        if schema is not None:
            self.schema = schema
        if status is not None:
            self.status = status
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self._record_version(change_note or "Updated")

    def _record_version(self, note: str) -> None:
        snapshot = {
            "version": self.version,
            "timestamp": self.updated_at,
            "note": note,
            "status": self.status.value,
            "metadata_hash": self._hash_metadata(),
        }
        self._version_history.append(snapshot)

    def _hash_metadata(self) -> str:
        import json
        raw = json.dumps(self.metadata, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @property
    def version_history(self) -> list[dict[str, Any]]:
        return list(self._version_history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "metadata": self.metadata,
            "schema": self.schema,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __repr__(self) -> str:
        return (
            f"CatalogEntry(id={self.asset_id!r}, name={self.name!r}, "
            f"type={self.asset_type.value}, v{self.version})"
        )


class MetadataCatalog:
    """
    Searchable registry of data assets with versioning.

    Supports registration, retrieval, search by tags/owner/type,
    and full version-history inspection.
    """

    def __init__(self) -> None:
        self._entries: dict[str, CatalogEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        asset_id: str,
        name: str,
        asset_type: AssetType,
        description: str = "",
        owner: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        schema: Optional[dict[str, str]] = None,
    ) -> CatalogEntry:
        """Register a new asset in the catalog."""
        if asset_id in self._entries:
            raise ValueError(
                f"Asset '{asset_id}' already registered. "
                f"Use update() to modify it."
            )
        entry = CatalogEntry(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type,
            description=description,
            owner=owner,
            tags=tags,
            metadata=metadata,
            schema=schema,
        )
        self._entries[asset_id] = entry
        logger.info("Registered asset: %s (%s)", name, asset_id)
        return entry

    def register_dataset(
        self,
        asset_id: str,
        name: str,
        description: str = "",
        owner: str = "",
        tags: Optional[list[str]] = None,
        schema: Optional[dict[str, str]] = None,
        **extra_metadata: Any,
    ) -> CatalogEntry:
        """Convenience method for registering a dataset."""
        return self.register(
            asset_id=asset_id,
            name=name,
            asset_type=AssetType.DATASET,
            description=description,
            owner=owner,
            tags=tags,
            metadata=extra_metadata or None,
            schema=schema,
        )

    def register_model(
        self,
        asset_id: str,
        name: str,
        description: str = "",
        owner: str = "",
        tags: Optional[list[str]] = None,
        **extra_metadata: Any,
    ) -> CatalogEntry:
        """Convenience method for registering an ML model."""
        return self.register(
            asset_id=asset_id,
            name=name,
            asset_type=AssetType.MODEL,
            description=description,
            owner=owner,
            tags=tags,
            metadata=extra_metadata or None,
        )

    def register_feature(
        self,
        asset_id: str,
        name: str,
        description: str = "",
        owner: str = "",
        tags: Optional[list[str]] = None,
        **extra_metadata: Any,
    ) -> CatalogEntry:
        """Convenience method for registering a feature."""
        return self.register(
            asset_id=asset_id,
            name=name,
            asset_type=AssetType.FEATURE,
            description=description,
            owner=owner,
            tags=tags,
            metadata=extra_metadata or None,
        )

    # ------------------------------------------------------------------
    # Retrieval and update
    # ------------------------------------------------------------------

    def get(self, asset_id: str) -> Optional[CatalogEntry]:
        return self._entries.get(asset_id)

    def update(self, asset_id: str, **kwargs: Any) -> CatalogEntry:
        entry = self._entries.get(asset_id)
        if entry is None:
            raise ValueError(f"Asset '{asset_id}' not found")
        entry.update(**kwargs)
        logger.info("Updated asset: %s (v%d)", asset_id, entry.version)
        return entry

    def deprecate(self, asset_id: str, reason: str = "") -> CatalogEntry:
        return self.update(
            asset_id,
            status=AssetStatus.DEPRECATED,
            change_note=f"Deprecated: {reason}" if reason else "Deprecated",
        )

    def archive(self, asset_id: str) -> CatalogEntry:
        return self.update(
            asset_id,
            status=AssetStatus.ARCHIVED,
            change_note="Archived",
        )

    def get_version_history(self, asset_id: str) -> list[dict[str, Any]]:
        entry = self._entries.get(asset_id)
        if entry is None:
            raise ValueError(f"Asset '{asset_id}' not found")
        return entry.version_history

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        owner: Optional[str] = None,
        asset_type: Optional[AssetType] = None,
        status: Optional[AssetStatus] = None,
    ) -> list[CatalogEntry]:
        """
        Search the catalog with optional filters.

        Args:
            query: Free-text search against name and description.
            tags: Filter entries that contain ALL specified tags.
            owner: Filter by owner (case-insensitive substring).
            asset_type: Filter by asset type.
            status: Filter by asset status.
        """
        results = list(self._entries.values())

        if asset_type:
            results = [e for e in results if e.asset_type == asset_type]
        if status:
            results = [e for e in results if e.status == status]
        if owner:
            owner_lower = owner.lower()
            results = [e for e in results if owner_lower in e.owner.lower()]
        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [
                e for e in results
                if tag_set.issubset({t.lower() for t in e.tags})
            ]
        if query:
            q = query.lower()
            results = [
                e for e in results
                if q in e.name.lower() or q in e.description.lower()
            ]

        return results

    def list_all(self) -> list[CatalogEntry]:
        return list(self._entries.values())

    def list_by_type(self, asset_type: AssetType) -> list[CatalogEntry]:
        return self.search(asset_type=asset_type)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for entry in self._entries.values():
            t = entry.asset_type.value
            s = entry.status.value
            by_type[t] = by_type.get(t, 0) + 1
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_assets": len(self._entries),
            "by_type": by_type,
            "by_status": by_status,
            "owners": sorted({e.owner for e in self._entries.values() if e.owner}),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._entries

    def __repr__(self) -> str:
        return f"MetadataCatalog(assets={len(self._entries)})"
