import os
import tempfile
import time
import unittest
import urllib.parse

from kwola.components.environments.PlaywrightBrowserSession import PlaywrightBrowserSession


class TestPlaywrightBrowserSession(unittest.TestCase):
    """Contract tests shared by the Chromium and Firefox Kwola labels."""

    def _session(self, browser):
        return PlaywrightBrowserSession(
            browser, headless=True, proxy_port=9, width=320, height=240,
            script_timeout=5, page_timeout=5,
        )

    def test_coordinate_session_contract_on_both_engines(self):
        for browser in ("chrome", "firefox"):
            with self.subTest(browser=browser):
                session = self._session(browser)
                try:
                    html = "<style>#b{position:absolute;left:10px;top:10px}#i{position:absolute;left:10px;top:60px}#scroll{height:2000px}</style><button id='b' onclick='console.error(\"expected\"); window.result=(window.result||0)+1' oncontextmenu='window.context=true;return false'>go</button><input id='i'><div id='scroll'></div>"
                    session.get("data:text/html," + urllib.parse.quote(html))
                    self.assertEqual(session.current_url.split(":", 1)[0], "data")
                    session.click_at(20, 20)
                    self.assertEqual(session.execute_script("return window.result;"), 1)
                    session.click_at(20, 20, count=2)
                    self.assertEqual(session.execute_script("return window.result;"), 3)
                    session.click_at(20, 20, button="right")
                    self.assertTrue(session.execute_script("return window.context;"))
                    self.assertIn("error", [entry["level"] for entry in session.consume_console_messages()])
                    session.type_at(20, 70, "hello")
                    self.assertEqual(session.execute_script("return document.querySelector('#i').value;"), "hello")
                    session.clear_at(20, 70)
                    self.assertEqual(session.execute_script("return document.querySelector('#i').value;"), "")
                    session.execute_script("document.body.style.height='2000px';")
                    session.scroll(200)
                    time.sleep(0.1)
                    self.assertGreater(session.execute_script("return window.scrollY;"), 0)
                    self.assertEqual(session.css_property_at(20, 20, "cursor"), "auto")
                    session.execute_script("alert('Kwola dialog');")
                    self.assertEqual(session.consume_dialogs()[0]["message"], "Kwola dialog")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image:
                        image_path = image.name
                    try:
                        session.save_screenshot(image_path)
                        self.assertGreater(os.path.getsize(image_path), 0)
                    finally:
                        os.unlink(image_path)
                finally:
                    session.quit()
                    session.quit()
