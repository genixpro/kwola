"""Runner lifecycle dispatch with observable best-effort failures."""

import logging
from collections.abc import Callable

from kwola.hooks import (
    HookExecutionError,
    HookFailure,
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
)

from .results import RunnerWarning


class RunnerLifecycle:
    def __init__(
        self,
        hooks: HookRegistry,
        run_id: str,
        clock: Callable[[], float],
    ) -> None:
        self._hooks = hooks
        self._run_id = run_id
        self._clock = clock
        self._warnings: list[RunnerWarning] = []
        self._logger = logging.getLogger(__name__)

    @property
    def warnings(self) -> tuple[RunnerWarning, ...]:
        return tuple(self._warnings)

    def dispatch(
        self,
        name: LifecycleEventName,
        subject_id: str | None = None,
        payload: tuple[tuple[str, object], ...] = (),
    ) -> None:
        failures = self._hooks.dispatch(
            LifecycleEvent(
                name=name,
                occurred_at=self._clock(),
                run_id=self._run_id,
                subject_id=subject_id,
                payload=payload,
            )
        )
        self._record(failures)

    def dispatch_preserving(
        self,
        name: LifecycleEventName,
        primary_error: BaseException | None,
        subject_id: str | None = None,
        payload: tuple[tuple[str, object], ...] = (),
    ) -> None:
        try:
            self.dispatch(name, subject_id, payload)
        except HookExecutionError as error:
            if primary_error is None:
                raise
            self._record((error.failure,))

    def finish(self, primary_error: BaseException | None) -> None:
        finish_error: HookExecutionError | None = None
        try:
            self.dispatch(LifecycleEventName.RUN_FINISHED)
        except HookExecutionError as error:
            self._record((error.failure,))
            if primary_error is None:
                finish_error = error
        finally:
            self._record(self._hooks.close())
        if finish_error is not None:
            raise finish_error

    def _record(self, failures: tuple[HookFailure, ...]) -> None:
        for failure in failures:
            warning = RunnerWarning(
                hook=failure.hook,
                event=failure.event,
                fatal=failure.fatal,
                error_type=failure.error_type,
                message=failure.message,
            )
            self._warnings.append(warning)
            self._logger.warning(
                "runner hook %s failed during %s (%s): %s",
                failure.hook,
                failure.event.value,
                failure.error_type,
                failure.message,
            )
