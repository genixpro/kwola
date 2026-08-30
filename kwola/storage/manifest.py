"""Small, inspectable run manifest."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

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

    @field_validator("file")
    @classmethod
    def checkpoint_path_is_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or "\\" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint file must be a confined relative POSIX path")
        return value

    @field_validator("sha256")
    @classmethod
    def checkpoint_digest_is_valid(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("checkpoint sha256 must contain 64 hexadecimal characters")
        return normalized


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
        schema_version: int = 1,
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
            schema_version=schema_version,
        )


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / MANIFEST_NAME).read_bytes())


def save_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    target = run_dir / MANIFEST_NAME
    _atomic_json_write(target, manifest.model_dump(mode="json"))
    return target
