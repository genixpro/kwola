"""Small, inspectable run manifest."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from kwola.config.io import _atomic_json_write
from kwola.config.models import ProfileName
from kwola.domain.actions import BrowserKind

MANIFEST_NAME = "manifest.json"


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointMetadata(ManifestModel):
    generation: int = Field(ge=0)
    file: str
    sha256: str
    published_at: float


class RunManifest(ManifestModel):
    schema_version: int = Field(default=1, ge=1)
    kwola_version: str
    target: AnyHttpUrl
    profile: ProfileName
    seed: int = Field(ge=0)
    enabled_browsers: tuple[BrowserKind, ...]
    checkpoint: CheckpointMetadata | None = None

    @classmethod
    def create(
        cls,
        *,
        target: AnyHttpUrl,
        profile: ProfileName,
        seed: int,
        enabled_browsers: tuple[BrowserKind, ...],
    ) -> "RunManifest":
        try:
            kwola_version = version("kwola")
        except PackageNotFoundError:
            kwola_version = "0+unknown"
        return cls(
            kwola_version=kwola_version,
            target=target,
            profile=profile,
            seed=seed,
            enabled_browsers=enabled_browsers,
        )


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / MANIFEST_NAME).read_bytes())


def save_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    target = run_dir / MANIFEST_NAME
    _atomic_json_write(target, manifest.model_dump(mode="json"))
    return target
