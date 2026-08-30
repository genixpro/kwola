"""Predictable lifecycle hook ordering and failure behavior."""

from dataclasses import dataclass
from typing import Protocol

from .events import LifecycleEvent, LifecycleEventName


class LifecycleHook(Protocol):
    name: str
    order: int
    fatal: bool
    events: frozenset[LifecycleEventName]

    def handle(self, event: LifecycleEvent) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HookFailure:
    hook: str
    event: LifecycleEventName
    fatal: bool
    error_type: str
    message: str


class HookExecutionError(RuntimeError):
    def __init__(self, failure: HookFailure) -> None:
        self.failure = failure
        super().__init__(
            f"hook {failure.hook!r} failed during {failure.event.value}: "
            f"{failure.error_type}: {failure.message}"
        )


class HookRegistry:
    def __init__(self, hooks: tuple[LifecycleHook, ...] = ()) -> None:
        names = [hook.name for hook in hooks]
        if len(names) != len(set(names)):
            raise ValueError("hook names must be unique")
        self._hooks = tuple(sorted(hooks, key=lambda hook: (hook.order, hook.name)))

    @property
    def hooks(self) -> tuple[LifecycleHook, ...]:
        return self._hooks

    def dispatch(self, event: LifecycleEvent) -> tuple[HookFailure, ...]:
        failures: list[HookFailure] = []
        for hook in self._hooks:
            if event.name not in hook.events:
                continue
            try:
                hook.handle(event)
            except Exception as error:
                failure = HookFailure(
                    hook=hook.name,
                    event=event.name,
                    fatal=hook.fatal,
                    error_type=type(error).__name__,
                    message=str(error),
                )
                failures.append(failure)
                if hook.fatal:
                    raise HookExecutionError(failure) from error
        return tuple(failures)

    def close(self) -> tuple[HookFailure, ...]:
        failures: list[HookFailure] = []
        for hook in reversed(self._hooks):
            try:
                hook.close()
            except Exception as error:
                failures.append(
                    HookFailure(
                        hook=hook.name,
                        event=LifecycleEventName.SHUTDOWN,
                        fatal=hook.fatal,
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )
        return tuple(failures)
