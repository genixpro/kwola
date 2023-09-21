#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium
import selenium.common
import selenium.webdriver.chrome.options
import time
import sys

def testChromedriver(verbose=True):
    """
        This command is used to test whether chomedriver is installed correctly.
    """

    targetURL = "http://kros3.kwola.io/"

    if verbose:
        print(f"Starting a Chrome browser through the chromedriver and pointing it at {targetURL}")

    chrome_options = selenium.webdriver.chrome.options.Options()
    chrome_options.headless = True
    chrome_options.add_argument(f"--no-sandbox")
    if sys.platform == "win32" or sys.platform == "win64":
        chrome_options.add_argument(f"--disable-dev-shm-usage")

    capabilities = selenium.webdriver.DesiredCapabilities.CHROME
    capabilities['loggingPrefs'] = {'browser': 'ALL'}

    driver = selenium.webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

    loginElement = None
    for attempt in range(5):
        driver.get(targetURL)
        try:
            loginElement = WebDriverWait(driver, 10, 0.25).until(
                EC.presence_of_element_located((By.CLASS_NAME, "btn-success"))
            )
            break
        except selenium.common.exceptions.NoSuchElementException:
            print(f"Diagnostic URL {targetURL} did not appear to load correctly. Received a no-such-element error. Waiting 10 seconds and then retrying.")
            time.sleep(10)
        except selenium.common.exceptions.TimeoutException:
            print(f"Diagnostic URL {targetURL} did not appear to load correctly. Received a timeout error. Waiting 10 seconds and then retrying.")
            time.sleep(10)

    driver.close()

    if loginElement is not None:
        if verbose:
            print(f"Congratulations! Your Selenium installation appears to be working. We were able to load {targetURL} with a headless browser.")
        return True
    else:
        if verbose:
            print(f"Unfortunately, your Selenium installation does not appear to be working. We were unable to load {targetURL} with the headless browser and confirm its the Google page.")
        return False


