"""Context-managed mitmproxy lifecycle."""

import asyncio
import socket
from contextlib import closing
from threading import Event, Thread
from typing import Self

from mitmproxy import options
from mitmproxy.tools.dump import DumpMaster

from .addon import InstrumentationAddon


class ProxyStartupError(RuntimeError):
    pass


class ProxyService:
    def __init__(self, addon: InstrumentationAddon, port: int = 0) -> None:
        self._addon = addon
        self._port = port or _free_port()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._master: DumpMaster | None = None
        self._thread: Thread | None = None
        self._ready = Event()
        self._stopped = Event()
        self._failure: BaseException | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def server(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self, timeout_seconds: float = 10.0) -> None:
        if self._thread is not None:
            raise RuntimeError("proxy is already started")
        self._ready.clear()
        self._stopped.clear()
        self._failure = None
        self._thread = Thread(target=self._run, name="kwola-proxy", daemon=True)
        self._thread.start()
        ready = self._ready.wait(timeout_seconds)
        if ready and self._failure is None:
            return
        detail = f": {self._failure}" if self._failure else ""
        try:
            self.close()
        except BaseException as cleanup_error:
            detail += f"; cleanup failed: {cleanup_error}"
        if not ready:
            raise ProxyStartupError(f"proxy did not start within {timeout_seconds}s{detail}")
        raise ProxyStartupError(f"proxy failed during startup{detail}")

    def close(self) -> None:
        master = self._master
        loop = self._loop
        thread = self._thread
        self._thread = None
        self._master = None
        self._loop = None
        failure: BaseException | None = None
        if master is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(master.shutdown)
            except BaseException as error:
                failure = error
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive() and failure is None:
                failure = ProxyStartupError("proxy thread did not stop within 5 seconds")
        if failure is not None:
            raise failure

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            proxy_options = options.Options(
                listen_host="127.0.0.1",
                listen_port=self._port,
                http2=False,
                ssl_insecure=True,
            )
            master = DumpMaster(proxy_options, loop=loop, with_termlog=False, with_dumper=False)
            self._master = master
            master.addons.add(self._addon)  # type: ignore[no-untyped-call]
            master.addons.add(_ReadyAddon(self._ready))  # type: ignore[no-untyped-call]
            loop.run_until_complete(master.run())
        except BaseException as error:
            self._failure = error
            self._ready.set()
        finally:
            self._stopped.set()
            _cancel_pending_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


class _ReadyAddon:
    def __init__(self, event: Event) -> None:
        self._event = event

    def running(self) -> None:
        self._event.set()


def _cancel_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
