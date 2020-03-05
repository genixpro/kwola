

from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.chrome.options import Options
import time
import numpy as np


prox = Proxy()
prox.proxy_type = ProxyType.MANUAL
prox.http_proxy = "localhost:8080"


chrome_options = Options()
# chrome_options.add_argument("--headless")


capabilities = webdriver.DesiredCapabilities.CHROME
prox.add_to_capabilities(capabilities)

driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

driver.get("http://172.17.0.2:3000/")



# The JavaScript that we want to inject. This will extract out the Kwola debug information.
injected_javascript = (
    'return window.kwolaCounters;'
)


lastValues = {}

while True:
    result = driver.execute_script(injected_javascript)

    for fileName, counterVector in result.items():
        if fileName in lastValues:
            lastCounterVector = lastValues[fileName]

            difference = np.array(counterVector) - np.array(lastCounterVector)

            if np.sum(difference) > 0:
                print(fileName, "has run!")

    lastValues = result

    time.sleep(10)



driver.quit()

