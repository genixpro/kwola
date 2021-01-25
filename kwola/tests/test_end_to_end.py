
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import KwolaCoreConfiguration
import shutil
import traceback
from ..config.logger import getLogger, setupLocalLogging

class TestEndToEnd(unittest.TestCase):
    def run_click_only_test(self, url):
        getLogger().info(f"\nStarting a click-only test targeting the URL {url}")

        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url=url,
                                                                        email="",
                                                                        password="",
                                                                        name="",
                                                                        paragraph="",
                                                                        enableTypeEmail=False,
                                                                        enableTypePassword=False,
                                                                        enableRandomNumberCommand=False,
                                                                        enableRandomBracketCommand=False,
                                                                        enableRandomMathCommand=False,
                                                                        enableRandomOtherSymbolCommand=False,
                                                                        enableDoubleClickCommand=False,
                                                                        enableRightClickCommand=False
                                                                        )

        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
            getLogger().info(f"Click-only test for URL {url} has completed successfully")
        except Exception:
            getLogger().error(f"Click-only test for URL {url} has failed. {traceback.format_exc()}")
            raise
        finally:
            shutil.rmtree(configDir)

    def test_amazon(self):
        self.run_click_only_test("https://amazon.com/")

    def test_apple(self):
        self.run_click_only_test("https://apple.com/")

    def test_bestbuy(self):
        self.run_click_only_test("https://bestbuy.com/")

    def test_bing(self):
        self.run_click_only_test("https://bing.com/")

    def test_britannica(self):
        self.run_click_only_test("https://britannica.com/")

    def test_businessinsider(self):
        self.run_click_only_test("https://businessinsider.com/")

    def test_cnet(self):
        self.run_click_only_test("https://cnet.com/")

    def test_cnn(self):
        self.run_click_only_test("https://cnn.com/")

    def test_craigslist(self):
        self.run_click_only_test("https://craigslist.org/")

    def test_dictionary(self):
        self.run_click_only_test("https://dictionary.com/")

    def test_ebay(self):
        self.run_click_only_test("https://ebay.com/")

    def test_etsy(self):
        self.run_click_only_test("https://etsy.com/")

    def test_forbes(self):
        self.run_click_only_test("https://forbes.com/")

    def test_foxnews(self):
        self.run_click_only_test("https://foxnews.com/")

    def test_gamepedia(self):
        self.run_click_only_test("https://gamepedia.com/")

    def test_homedepot(self):
        self.run_click_only_test("https://homedepot.com/")

    def test_imdb(self):
        self.run_click_only_test("https://imdb.com/")

    def test_indeed(self):
        self.run_click_only_test("https://indeed.com/")

    def test_linkdin(self):
        self.run_click_only_test("https://linkedin.com/")

    def test_mayoclinic(self):
        self.run_click_only_test("https://mayoclinic.org/")

    def test_mapquest(self):
        self.run_click_only_test("https://mapquest.com/")

    def test_merriam_webster(self):
        self.run_click_only_test("https://merriam-webster.com/")

    def test_microsoft(self):
        self.run_click_only_test("https://microsoft.com/")

    def test_netflix(self):
        self.run_click_only_test("https://netflix.com/")

    def test_nih(self):
        self.run_click_only_test("https://nih.gov/")

    def test_nytimes(self):
        self.run_click_only_test("https://nytimes.com/")

    def test_quora(self):
        self.run_click_only_test("https://quora.com/")

    def test_reddit(self):
        self.run_click_only_test("https://reddit.com/")

    def test_rotten_tomatoes(self):
        self.run_click_only_test("https://rottentomatoes.com/")

    def test_target(self):
        self.run_click_only_test("https://target.com/")

    def test_twitter(self):
        self.run_click_only_test("https://twitter.com/")

    def test_walmart(self):
        self.run_click_only_test("https://walmart.com/")

    def test_weather(self):
        self.run_click_only_test("https://weather.com/")

    def test_webmd(self):
        self.run_click_only_test("https://webmd.com/")

    def test_wikipedia(self):
        self.run_click_only_test("https://wikipedia.org/")

    def test_google(self):
        self.run_click_only_test("https://www.google.com/")

    def test_yahoo(self):
        self.run_click_only_test("https://www.yahoo.com/")

    def test_youtube(self):
        self.run_click_only_test("https://www.youtube.com/")

    def test_yelp(self):
        self.run_click_only_test("https://yelp.com/")

    def test_zillow(self):
        self.run_click_only_test("https://zillow.com/")

