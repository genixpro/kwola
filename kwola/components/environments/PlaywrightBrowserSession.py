"""Small synchronous Playwright boundary used by Kwola web sessions.

Kwola intentionally keeps coordinate based actions.  This adapter is therefore
not a selector-test abstraction: it exposes pages, pixels and JavaScript in the
same terms as the former WebDriver boundary while keeping Playwright details out
of plugins and environment code.
"""
from playwright.sync_api import sync_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError


class BrowserSessionError(RuntimeError):
    pass


class PlaywrightBrowserSession:
    def __init__(self, browser_name, *, headless, proxy_port, width, height,
                 script_timeout, page_timeout, cache_dir=None):
        if browser_name not in ("chrome", "firefox"):
            raise ValueError("Unsupported browser %r. Valid values are 'chrome' and 'firefox'." % browser_name)
        self.browser_name = browser_name
        self.browser = None
        self.context = None
        self.page = None
        self._closed = False
        self._playwright = sync_playwright().start()
        browser_type = self._playwright.chromium if browser_name == "chrome" else self._playwright.firefox
        launch_args = ["--no-sandbox"] if browser_name == "chrome" else []
        if cache_dir and browser_name == "chrome":
            launch_args.extend(["--disk-cache-dir=%s" % cache_dir, "--disk-cache-size=1073741824"])
        try:
            self.browser = browser_type.launch(headless=headless, proxy={"server": "http://127.0.0.1:%s" % proxy_port}, args=launch_args)
            self.context = self.browser.new_context(viewport={"width": width, "height": height})
            self.context.set_default_timeout(int(script_timeout * 1000))
            self.context.set_default_navigation_timeout(int(page_timeout * 1000))
            self.page = self.context.new_page()
        except (PlaywrightError, RuntimeError) as exc:
            self.quit()
            raise BrowserSessionError(
                "Unable to start Playwright %s. Run `playwright install chromium firefox`; "
                "if this target uses HTTPS instrumentation, install the Kwola proxy certificate first. %s" % (browser_name, exc)
            ) from exc
        self._console_messages = []
        self._dialogs = []
        self.page.on("console", lambda message: self._console_messages.append({"level": message.type, "message": message.text}))
        self.page.on("dialog", self._on_dialog)

    def _on_dialog(self, dialog):
        self._dialogs.append({"type": dialog.type, "message": dialog.message})
        dialog.accept()

    @property
    def current_url(self):
        return self.page.url

    @property
    def page_source(self):
        return self.page.content()

    def get(self, url):
        self.page.goto(url, wait_until="domcontentloaded")

    def execute_script(self, script, *args):
        # A normal function (rather than an arrow function) intentionally
        # preserves Selenium's `arguments[n]` convention used by old plugins.
        return self.page.evaluate("(args) => (function() { " + script + " }).apply(null, args)", list(args))

    def css_property_at(self, x, y, property_name):
        """Return a computed CSS property at viewport coordinates."""
        return self.page.evaluate(
            "([x, y, propertyName]) => { const element = document.elementFromPoint(x, y); "
            "return element ? getComputedStyle(element).getPropertyValue(propertyName) : null; }",
            [x, y, property_name],
        )

    def save_screenshot(self, path):
        self.page.screenshot(path=path)

    def get_screenshot_as_png(self):
        return self.page.screenshot()

    def get_window_rect(self):
        size = self.page.viewport_size
        return {"x": 0, "y": 0, "width": size["width"], "height": size["height"]}

    def element_at(self, x, y):
        return self.page.locator("body"), x, y

    def click_at(self, x, y, *, count=1, button="left", delay=0):
        self.page.mouse.click(x, y, button=button, click_count=count, delay=delay)

    def type_at(self, x, y, text, *, delay=0):
        self.page.mouse.click(x, y)
        self.page.keyboard.type(text, delay=delay)

    def clear_at(self, x, y):
        self.page.mouse.click(x, y)
        self.page.keyboard.press("ControlOrMeta+A")
        self.page.keyboard.press("Backspace")

    def scroll(self, amount):
        self.page.mouse.wheel(0, amount)

    def consume_console_messages(self):
        messages, self._console_messages = self._console_messages, []
        return messages

    def consume_dialogs(self):
        dialogs, self._dialogs = self._dialogs, []
        return dialogs

    def is_installed(self):
        # `executable_path` is supplied by the pinned Playwright package and
        # verifies the browser payload without relying on system WebDrivers.
        browser_type = self._playwright.chromium if self.browser_name == "chrome" else self._playwright.firefox
        return browser_type.executable_path

    def quit(self):
        """Close page, context, browser and Playwright in one idempotent path."""
        if self._closed:
            return
        self._closed = True
        for resource in (self.page, self.context, self.browser):
            if resource is not None:
                try:
                    resource.close()
                except (PlaywrightError, RuntimeError):
                    # A browser-death recovery may have closed an outer
                    # resource already; shutdown must remain best effort.
                    pass
        try:
            self._playwright.stop()
        except (PlaywrightError, RuntimeError):
            pass
        finally:
            self.page = None
            self.context = None
            self.browser = None
