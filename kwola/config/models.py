"""Pydantic configuration models used at run and process boundaries."""

import os
import re
from typing import Literal, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from kwola.domain.actions import BrowserKind

ProfileName = Literal["testing", "standard", "rig"]
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _environment_secret(name: str | None, label: str) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{label} environment variable is unset or empty: {name}")
    return value


def _valid_environment_name(value: str | None) -> str | None:
    if value is not None and _ENVIRONMENT_NAME.fullmatch(value) is None:
        raise ValueError("credential environment variable names must be shell identifiers")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ViewportConfig(StrictModel):
    width: int = Field(ge=320, le=7680)
    height: int = Field(ge=240, le=4320)


class LoginConfig(StrictModel):
    enabled: bool = False
    email: str | None = Field(default=None, exclude=True, repr=False)
    password: str | None = Field(default=None, exclude=True, repr=False)
    email_environment: str | None = None
    password_environment: str | None = None

    _environment_names_are_valid = field_validator("email_environment", "password_environment")(
        _valid_environment_name
    )

    @model_validator(mode="after")
    def credentials_required_when_enabled(self) -> Self:
        email_source = self.email or self.email_environment
        password_source = self.password or self.password_environment
        if self.enabled and (not email_source or not password_source):
            raise ValueError("autologin requires email and password credential sources")
        return self

    def credentials(self) -> tuple[str, str]:
        email = self.email or _environment_secret(self.email_environment, "autologin email")
        password = self.password or _environment_secret(
            self.password_environment, "autologin password"
        )
        if not email or not password:
            raise ValueError("autologin credentials are unavailable")
        return email, password


class BrowserConfig(StrictModel):
    enabled: tuple[BrowserKind, ...] = (BrowserKind.CHROMIUM, BrowserKind.FIREFOX)
    viewports: tuple[ViewportConfig, ...] = (ViewportConfig(width=1920, height=1080),)
    headless: bool = True
    prevent_offsite_navigation: bool = True
    allowed_navigation_origins: tuple[AnyHttpUrl, ...] = ()
    page_load_timeout_seconds: float = Field(default=60.0, gt=0)
    action_timeout_seconds: float = Field(default=15.0, gt=0)
    network_idle_seconds: float = Field(default=0.5, ge=0)
    action_settle_seconds: float = Field(default=0.25, ge=0)
    autologin: LoginConfig = LoginConfig()

    @model_validator(mode="after")
    def browser_matrix_is_nonempty(self) -> Self:
        if not self.enabled:
            raise ValueError("at least one browser must be enabled")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError("enabled browsers must be unique")
        if not self.viewports:
            raise ValueError("at least one viewport must be configured")
        for origin in self.allowed_navigation_origins:
            if origin.username or origin.password:
                raise ValueError("allowed navigation origins cannot contain credentials")
            if origin.path not in {None, "", "/"} or origin.query or origin.fragment:
                raise ValueError(
                    "allowed navigation origins cannot contain a path, query, or fragment"
                )
        return self


class InstrumentationConfig(StrictModel):
    enabled: bool = True
    proxy_port: int = Field(default=0, ge=0, le=65535)
    rewrite_javascript: bool = True
    rewrite_html: bool = True
    capture_console: bool = True
    capture_network: bool = True
    capture_html: bool = False
    capture_resources: bool = True
    branch_trace_timeout_seconds: float = Field(default=30.0, gt=0)


class ExplorationAxisConfig(StrictModel):
    start_random_rate: float = Field(default=1.0, ge=0, le=1)
    end_random_rate: float = Field(default=1.0, ge=0, le=1)
    start_weighted_random_rate: float = Field(default=1.0, ge=0, le=1)
    end_weighted_random_rate: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def weighted_exploration_is_within_total(self) -> Self:
        if self.start_weighted_random_rate > self.start_random_rate:
            raise ValueError("start weighted-random rate cannot exceed total random rate")
        if self.end_weighted_random_rate > self.end_random_rate:
            raise ValueError("end weighted-random rate cannot exceed total random rate")
        return self


class ExplorationConfig(StrictModel):
    action: ExplorationAxisConfig = ExplorationAxisConfig(
        start_random_rate=0.4,
        end_random_rate=1.0,
        start_weighted_random_rate=0.4,
        end_weighted_random_rate=1.0,
    )
    session: ExplorationAxisConfig = ExplorationAxisConfig()
    test_step: ExplorationAxisConfig = ExplorationAxisConfig(
        start_random_rate=0.01,
        end_random_rate=0.8,
        start_weighted_random_rate=0.01,
        end_weighted_random_rate=0.8,
    )
    max_test_step_index: int = Field(default=1000, ge=2)


class RewardConfig(StrictModel):
    impossible_action: float = -10.0
    code_executed: float = 0.001
    no_code_executed: float = -0.01
    new_code_executed: float = 0.3
    no_new_code_executed: float = 0.0
    network_traffic: float = 0.005
    no_network_traffic: float = 0.0
    new_network_traffic: float = 0.05
    no_new_network_traffic: float = 0.0
    screenshot_changed: float = 0.001
    no_screenshot_change: float = -0.01
    new_screenshot: float = 0.0
    no_new_screenshot: float = 0.0
    url_changed: float = 0.001
    no_url_change: float = 0.0
    new_url: float = 0.1
    no_new_url: float = 0.0
    log_output: float = 0.001
    no_log_output: float = 0.0
    discount_rate: float = Field(default=0.85, gt=0, le=1)
    max_discounted_reward: float = Field(default=10.0, gt=0)


class ActionWeightsConfig(StrictModel):
    click: float = Field(default=1.0, gt=0)
    clear: float = Field(default=0.5, gt=0)
    custom_type: float = Field(default=0.5, gt=0)
    double_click: float = Field(default=0.2, gt=0)
    right_click: float = Field(default=0.2, gt=0)
    scrolling: float = Field(default=0.2, gt=0)
    type_brackets: float = Field(default=0.3, gt=0)
    type_email: float = Field(default=1.0, gt=0)
    type_math: float = Field(default=0.3, gt=0)
    type_name: float = Field(default=0.7, gt=0)
    type_number: float = Field(default=1.0, gt=0)
    type_other_symbol: float = Field(default=0.3, gt=0)
    type_paragraph: float = Field(default=0.7, gt=0)
    type_password: float = Field(default=1.0, gt=0)
    random_generated: float = Field(default=0.4, gt=0)
    random_letters: float = Field(default=1.0, gt=0)


class ActionConfig(StrictModel):
    email: str | None = Field(default=None, exclude=True, repr=False)
    password: str | None = Field(default=None, exclude=True, repr=False)
    email_environment: str | None = None
    password_environment: str | None = None
    name: str | None = None
    paragraph: str | None = None
    random_letters: bool = False
    random_address: bool = False
    random_email: bool = True
    random_phone_number: bool = False
    random_paragraph: bool = False
    random_date_time: bool = False
    random_credit_card: bool = False
    random_url: bool = False
    random_number: bool = True
    random_brackets: bool = False
    random_math: bool = False
    random_other_symbol: bool = False
    double_click: bool = False
    right_click: bool = False
    scrolling: bool = True
    weights: ActionWeightsConfig = ActionWeightsConfig()

    _environment_names_are_valid = field_validator("email_environment", "password_environment")(
        _valid_environment_name
    )

    def resolved_email(self) -> str | None:
        return self.email or _environment_secret(self.email_environment, "action email")

    def resolved_password(self) -> str | None:
        return self.password or _environment_secret(self.password_environment, "action password")


class PolicyConfig(StrictModel):
    exploration: ExplorationConfig = ExplorationConfig()
    rewards: RewardConfig = RewardConfig()
    repeat_action_override: bool = True
    max_repeat_maps_without_new_branches: int = Field(default=3, ge=0)
    testing_sequence_length: int = Field(default=5, ge=2)
    custom_typing_strings: tuple[str, ...] = ()
    actions: ActionConfig = ActionConfig()


class ConvolutionLayerConfig(StrictModel):
    kernels: int = Field(ge=1)
    kernel_size: int = Field(default=3, ge=1)
    stride: int = Field(default=1, ge=1)
    padding: int = Field(default=1, ge=0)
    dilation: int = Field(default=1, ge=1)


class ModelConfig(StrictModel):
    image_downscale_ratio: float = Field(default=0.3, gt=0, le=1)
    layers: tuple[ConvolutionLayerConfig, ...]
    pixel_features: int = Field(ge=1)
    recent_action_features: int = Field(default=16, ge=1)
    recent_action_history: int = Field(default=5, ge=1)
    additional_stamp_depth: int = Field(default=5, ge=1)
    additional_stamp_edge: int = Field(default=2, ge=1)
    symbol_dictionary_size: int = Field(default=25_000, ge=1)
    symbol_embedding_size: int = Field(default=32, ge=1)
    enable_cursor_prediction: bool = True
    enable_execution_feature_prediction: bool = True
    enable_trace_prediction: bool = True
    prediction_head_kernel_size: int = Field(default=3, ge=1)
    prediction_head_stride: int = Field(default=1, ge=1)
    prediction_head_padding: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def topology_has_five_layers(self) -> Self:
        if len(self.layers) != 5:
            raise ValueError("TraceNet requires exactly five convolutional layers")
        if self.layers[3].kernels != self.pixel_features:
            raise ValueError("TraceNet layer four kernels must match pixel_features")
        return self


class LossConfig(StrictModel):
    cursor_prediction: float = Field(default=1.0, ge=0)
    execution_feature: float = Field(default=1.0, ge=0)
    execution_trace: float = Field(default=1.0, ge=0)
    discounted_future_reward: float = Field(default=8.0, ge=0)
    present_reward: float = Field(default=16.0, ge=0)


class TrainingConfig(StrictModel):
    batch_size: int = Field(default=4, ge=1)
    batches_per_iteration: int = Field(default=8, ge=1)
    min_batches_per_iteration: int = Field(default=1, ge=1)
    max_batches_per_iteration: int = Field(default=1200, ge=1)
    batch_iteration_adjustment: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: Literal["adam", "adamax"] = "adamax"
    gradient_beta: float = Field(default=0.97, gt=0, lt=1)
    squared_gradient_beta: float = Field(default=0.999, gt=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_clip_norm: float = Field(default=10.0, gt=0)
    device_indices: tuple[int, ...] = ()
    world_size: int = Field(default=1, ge=1)
    sample_cache_workers: int = Field(default=4, ge=0)
    sample_cache_version: int = Field(default=1, ge=1)
    use_shared_memory_spool: bool = True
    cpu_threads_per_rank: int = Field(default=0, ge=0, le=32)
    batch_prefetch: bool = False
    decoded_image_cache_size: int = Field(default=0, ge=0)
    telemetry_every_iterations: int = Field(default=10, ge=1)
    checkpoint_every_iterations: int = Field(default=1, ge=1)
    target_network_update_every: int = Field(default=250, ge=1)
    crop_width: int = Field(default=320, ge=8)
    crop_height: int = Field(default=320, ge=8)
    next_crop_width: int = Field(default=448, ge=8)
    next_crop_height: int = Field(default=448, ge=8)
    crop_random_x: int = Field(default=100, ge=0)
    crop_random_y: int = Field(default=100, ge=0)
    recent_action_image_radius: int = Field(default=40, ge=1)
    recent_action_image_decay: float = Field(default=0.8, gt=0, le=1)
    losses: LossConfig = LossConfig()

    @model_validator(mode="after")
    def distributed_settings_are_consistent(self) -> Self:
        if self.device_indices and self.world_size != len(self.device_indices):
            raise ValueError("world_size must match the number of device indices")
        if not self.device_indices and self.world_size != 1:
            raise ValueError("CPU training only supports world_size=1")
        dimensions = (
            self.crop_width,
            self.crop_height,
            self.next_crop_width,
            self.next_crop_height,
        )
        if any(value % 8 for value in dimensions):
            raise ValueError("training crop dimensions must be divisible by 8")
        if (
            not self.min_batches_per_iteration
            <= self.batches_per_iteration
            <= self.max_batches_per_iteration
        ):
            raise ValueError("batches_per_iteration must be within its adaptive bounds")
        return self


class StorageConfig(StrictModel):
    database_map_size_bytes: int = Field(default=4 * 1024**3, ge=1024**2)
    codec_compression_level: int = Field(default=3, ge=-7, le=22)
    blobs_directory: str = "blobs"
    database_directory: str = "run.lmdb"
    cache_directory: str = "cache"
    checkpoints_directory: str = "checkpoints"


class ReportingConfig(StrictModel):
    charts: bool = True
    debug_videos: bool = True
    annotated_videos: bool = True
    bug_reports: bool = True
    chart_every_testing_steps: int = Field(default=5, ge=1)
    debug_video_every_testing_steps: int = Field(default=5, ge=1)
    debug_video_frames_per_second: float = Field(default=2.0, gt=0)
    debug_video_map_downscale: int = Field(default=8, ge=1)
    video_timeout_seconds: float = Field(default=900.0, gt=0)


class OrchestrationConfig(StrictModel):
    browser_workers: int = Field(default=1, ge=1, le=64)
    browser_cpu_threads: int = Field(default=1, ge=1, le=32)
    browser_max_consecutive_failures: int = Field(default=5, ge=1)
    browser_retry_base_seconds: float = Field(default=1.0, gt=0)
    browser_retry_max_seconds: float = Field(default=60.0, gt=0)
    worker_timeout_seconds: float = Field(default=3600.0, gt=0)
    telemetry_interval_seconds: float = Field(default=5.0, gt=0)
    minimum_traces_before_training: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def retry_delays_are_consistent(self) -> Self:
        if self.browser_retry_max_seconds < self.browser_retry_base_seconds:
            raise ValueError("browser retry maximum must be at least the base delay")
        return self


class RunConfig(StrictModel):
    schema_version: int = Field(default=2, ge=2, le=2)
    target: AnyHttpUrl
    profile: ProfileName
    seed: int = Field(ge=0, le=2**63 - 1)
    browser: BrowserConfig
    instrumentation: InstrumentationConfig
    policy: PolicyConfig
    model: ModelConfig
    training: TrainingConfig
    storage: StorageConfig
    reporting: ReportingConfig
    orchestration: OrchestrationConfig = OrchestrationConfig()

    @model_validator(mode="after")
    def instrumentation_requirements_are_consistent(self) -> Self:
        if not self.instrumentation.enabled and self.instrumentation.rewrite_javascript:
            raise ValueError("JavaScript rewriting requires instrumentation")
        return self
