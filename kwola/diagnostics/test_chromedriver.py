#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium
import selenium.common
import selenium.webdriver.chrome.options
import time

def testChromedriver(verbose=True):
    """
        This command is used to test whether chomedriver is installed correctly.
    """

    targetURL = "https://google.com/"

    if verbose:
        print(f"Starting a Chrome browser through the chromedriver and pointing it at {targetURL}")

    chrome_options = selenium.webdriver.chrome.options.Options()
    chrome_options.headless = True
    chrome_options.add_argument(f"--no-sandbox")

    capabilities = selenium.webdriver.DesiredCapabilities.CHROME
    capabilities['loggingPrefs'] = {'browser': 'ALL'}

    driver = selenium.webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

    driver.get(targetURL)

    googleBodyElement = None
    try:
        googleBodyElement = WebDriverWait(driver, 10, 0.25).until(
            EC.presence_of_element_located((By.ID, "gsr"))
        )
    except selenium.common.exceptions.NoSuchElementException:
        pass

    driver.close()

    if googleBodyElement is not None:
        if verbose:
            print(f"Congratulations! Your Selenium installation appears to be working. We were able to load {targetURL} with a headless browser.")
        return True
    else:
        if verbose:
            print(f"Unfortunately, your Selenium installation does not appear to be working. We were unable to load {targetURL} with the headless browser and confirm its the Google page.")
        return False


