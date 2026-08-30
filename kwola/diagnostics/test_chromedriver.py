"""Playwright browser diagnostic (keeps the historical module/CLI name)."""

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def _test_browser(browser_name, verbose=True):
    with sync_playwright() as playwright:
        browser_type = playwright.chromium if browser_name == "chrome" else playwright.firefox
        executable = browser_type.executable_path
        if not executable:
            raise RuntimeError("The pinned Playwright %s payload is absent. Run `uv run playwright install chromium firefox`." % browser_name)
        browser = browser_type.launch(headless=True, args=["--no-sandbox"] if browser_name == "chrome" else [])
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.set_content("<button class='kwola-browser-diagnostic'>Kwola</button>")
            page.locator(".kwola-browser-diagnostic").click()
            if verbose:
                print("Playwright %s launched successfully (%s)." % (browser_name, executable))
            return True
        finally:
            browser.close()


def testChromedriver(verbose=True):
    """Deprecated name retained for integrations; tests Chromium and Firefox."""
    success = True
    for browser_name in ("chrome", "firefox"):
        try:
            success = _test_browser(browser_name, verbose) and success
        except (PlaywrightError, RuntimeError) as exc:
            success = False
            if verbose:
                print("Playwright %s diagnostic failed: %s" % (browser_name, exc))
    if verbose and not success:
        print("Install the pinned browsers with: uv run playwright install chromium firefox")
    return success


testBrowser = testChromedriver
