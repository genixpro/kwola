"""Pydantic configuration models used at run and process boundaries."""

from typing import Literal, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from kwola.domain.actions import BrowserKind

ProfileName = Literal["testing", "standard"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ViewportConfig(StrictModel):
    width: int = Field(ge=320, le=7680)
    height: int = Field(ge=240, le=4320)


class LoginConfig(StrictModel):
    enabled: bool = False
    email: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def credentials_required_when_enabled(self) -> Self:
        if self.enabled and (not self.email or not self.password):
            raise ValueError("autologin requires both email and password")
        return self


class BrowserConfig(StrictModel):
    enabled: tuple[BrowserKind, ...] = (BrowserKind.CHROMIUM,)
    viewports: tuple[ViewportConfig, ...] = (ViewportConfig(width=1920, height=1080),)
    headless: bool = True
    prevent_offsite_navigation: bool = True
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
    capture_screenshots: bool = True
    branch_trace_timeout_seconds: float = Field(default=30.0, gt=0)


class ExplorationAxisConfig(StrictModel):
    start_random_rate: float = Field(default=1.0, ge=0, le=1)
    end_random_rate: float = Field(default=1.0, ge=0, le=1)
    start_weighted_random_rate: float = Field(default=1.0, ge=0, le=1)
    end_weighted_random_rate: float = Field(default=1.0, ge=0, le=1)


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
    action_failure: float = -0.02
    action_success: float = 0.0
    impossible_action: float = -10.0
    code_executed: float = 0.001
    no_code_executed: float = -0.01
    new_code_executed: float = 0.3
    no_new_code_executed: float = 0.0
    code_prevalence_exponential_base: float = 2.718
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


class PolicyConfig(StrictModel):
    exploration: ExplorationConfig = ExplorationConfig()
    rewards: RewardConfig = RewardConfig()
    weighted_random_actions: bool = True
    repeat_action_override: bool = True
    max_repeat_maps_without_new_branches: int = Field(default=3, ge=0)
    testing_sequence_length: int = Field(default=5, ge=2)
    custom_typing_strings: tuple[str, ...] = ()


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
    action_probability: float = Field(default=0.02, ge=0)
    advantage: float = Field(default=2.0, ge=0)
    cursor_prediction: float = Field(default=1.0, ge=0)
    execution_feature: float = Field(default=1.0, ge=0)
    execution_trace: float = Field(default=1.0, ge=0)
    discounted_future_reward: float = Field(default=8.0, ge=0)
    present_reward: float = Field(default=16.0, ge=0)
    state_value: float = Field(default=0.1, ge=0)


class TrainingConfig(StrictModel):
    batch_size: int = Field(default=4, ge=1)
    batches_per_iteration: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: Literal["adam", "adamax"] = "adamax"
    gradient_beta: float = Field(default=0.97, gt=0, lt=1)
    squared_gradient_beta: float = Field(default=0.999, gt=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    device_indices: tuple[int, ...] = ()
    world_size: int = Field(default=1, ge=1)
    sample_cache_workers: int = Field(default=4, ge=0)
    sample_cache_version: int = Field(default=1, ge=1)
    use_shared_memory_spool: bool = True
    checkpoint_every_iterations: int = Field(default=1, ge=1)
    losses: LossConfig = LossConfig()

    @model_validator(mode="after")
    def distributed_settings_are_consistent(self) -> Self:
        if self.device_indices and self.world_size != len(self.device_indices):
            raise ValueError("world_size must match the number of device indices")
        if not self.device_indices and self.world_size != 1:
            raise ValueError("CPU training only supports world_size=1")
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
    video_timeout_seconds: float = Field(default=900.0, gt=0)


class RunConfig(StrictModel):
    schema_version: int = Field(default=1, ge=1)
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

    @model_validator(mode="after")
    def instrumentation_requirements_are_consistent(self) -> Self:
        if not self.instrumentation.enabled and self.instrumentation.rewrite_javascript:
            raise ValueError("JavaScript rewriting requires instrumentation")
        return self
