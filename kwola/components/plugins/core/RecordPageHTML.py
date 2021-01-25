from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import os
from kwola.config.logger import getLogger
import re
from bs4 import BeautifulSoup
import urllib.parse
import urllib3
import requests
import traceback
import bz2
import selenium.common.exceptions



class RecordPageHTML(WebEnvironmentPluginBase):
    def __init__(self, config):
        self.config = config
        self.cssUrlRegex = re.compile(b"url\\s*\\(\\s*['\"]([^\\)'\"]+)['\"]\\s*\\)")
        self.failedResourceUrls = set()

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        pass


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        pass


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        self.saveHTML(webDriver, proxy, executionTrace)

    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass

    def cleanup(self, webDriver, proxy, executionSession):
        pass

    def saveHTML(self, webDriver, proxy, executionTrace):
        try:

            def modifyHTML(data, baseURL):
                soup = BeautifulSoup(data, features="html.parser")

                base = soup.find('base')
                if base is not None and 'href' in base.attrs:
                    baseURL = urllib.parse.urljoin(baseURL, base['href'])

                for jsCodeElement in soup.find_all('script', recursive=True):
                    jsCodeElement.extract()

                for baseTagElement in soup.find_all('base', recursive=True):
                    baseTagElement.extract()

                for elementIndex, element in enumerate(soup.find_all('link', recursive=True)):
                    if 'rel' in element.attrs and element['rel'][0] == 'stylesheet':
                        modifyElementResourceReference(element, baseURL, modifyCSS)
                    else:
                        modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('img', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('video', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('audio', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('source', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('track', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('embed', recursive=True)):
                    modifyElementResourceReference(element, baseURL)

                for elementIndex, element in enumerate(soup.find_all('iframe', recursive=True)):
                    modifyElementResourceReference(element, baseURL, modifyHTML)

                return str(soup.prettify())

            def modifyElementResourceReference(element, baseURL, resourceDataModifyFunc=None):
                attributeForResource = None
                if 'href' in element.attrs:
                    attributeForResource = 'href'
                elif 'src' in element.attrs:
                    attributeForResource = 'src'

                if attributeForResource is not None and element[attributeForResource]:
                    resourceURL = element[attributeForResource]
                    newURL = getResourceLink(resourceURL, baseURL, resourceDataModifyFunc)
                    if newURL is not None:
                        element[attributeForResource] = newURL
                    else:
                        element[attributeForResource] = "data:,"

            def urlWithoutFragment(url):
                parsed = list(urllib.parse.urlparse(url))
                parsed[5] = ""
                return urllib.parse.urlunparse(parsed)

            def getResourceLink(resourceURL, baseURL, resourceDataModifyFunc=None):
                resourceURL = urlWithoutFragment(urllib.parse.urljoin(baseURL, resourceURL))

                versionId, data = proxy.getResourceVersion(resourceURL)
                if versionId is None:
                    try:
                        proxies = {
                            'http': f'http://127.0.0.1:{proxy.port}',
                            'https': f'http://127.0.0.1:{proxy.port}',
                        }
                        response = requests.get(resourceURL, proxies=proxies, verify=False)
                        versionId, data = proxy.getResourceVersion(resourceURL)
                    except requests.exceptions.RequestException:
                        pass

                if data is None:
                    if resourceURL not in self.failedResourceUrls:
                        getLogger().warning(f"WARNING! Failed to grab the resource at URL: {resourceURL}")
                        self.failedResourceUrls.add(resourceURL)
                    return None
                else:
                    # urlPath = urllib.parse.urlparse(resourceURL).path
                    # extension = os.path.splitext(urlPath)[1]

                    # if resourceDataModifyFunc is not None:
                    #     data = resourceDataModifyFunc(data, resourceURL)

                    return f"kwolaResourceVersion://{versionId}"

            def modifyCSS(data, baseURL):
                matches = self.cssUrlRegex.findall(data)
                for match in matches:
                    resourceURL = match
                    newURL = getResourceLink(str(resourceURL, 'utf8'), baseURL)
                    if newURL is not None:
                        data = data.replace(resourceURL, bytes(newURL, 'utf8'))

                return data

            webDriver.execute_script("""
                function uniques(a)
                {
                    var seen = {};
                    return a.filter(function(item) {
                        return seen.hasOwnProperty(item) ? false : (seen[item] = true);
                    });
                }

                const domElements = document.querySelectorAll("*");
                for(let element of domElements)
                {
                    if (element.tagName === "INPUT")
                    {
                        element.setAttribute("value", element.value);
                    }
                    
                    const bounds = element.getBoundingClientRect();

                    element.setAttribute("data-kwola-left", bounds.left);
                    element.setAttribute("data-kwola-top", bounds.top);
                    element.setAttribute("data-kwola-bottom", bounds.bottom);
                    element.setAttribute("data-kwola-right", bounds.right);

                    if (window.kwolaEvents && window.kwolaEvents.has(element))
                    {
                        element.setAttribute("data-kwola-event-handlers", window.kwolaEvents.get(element).toString());
                    }

                    const elementAtPosition = document.elementFromPoint(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
                    if (elementAtPosition === null || element.contains(elementAtPosition) || elementAtPosition.contains(element))
                    {
                        element.setAttribute("data-kwola-is-on-top", "true");
                    }
                    else
                    {
                        element.setAttribute("data-kwola-is-on-top", "false");
                    }

                    const isVisible = !!( element.offsetWidth || element.offsetHeight || element.getClientRects().length );
                    element.setAttribute("data-kwola-is-visible", isVisible.toString());

                    element.setAttribute("data-kwola-scroll-top", element.scrollTop);
                    if (element.scrollTop)
                    {
                        let onload = element.getAttribute("onload");

                        if (!onload)
                        {
                            onload = "";
                        }               
                        onload = `this.scrollTop = ${element.scrollTop}; ${onload}`;
                             
                        element.setAttribute("onload", onload);
                    }
                }
            """)

            pageUrl = webDriver.current_url
            source = str(webDriver.page_source)

            # getLogger().info(f"Saving HTML of page {pageUrl} for execution trace {executionTrace.id}")

            newHtml = modifyHTML(source, pageUrl)
            newHtmlBytes = bytes(newHtml, 'utf8')
            compressed = bz2.compress(newHtmlBytes)

            self.config.saveKwolaFileData("saved_pages", executionTrace.id + ".html", compressed)

            return None
        except urllib3.exceptions.MaxRetryError:
            self.hasBrowserDied = True
            self.browserDeathReason = f"Following fatal error occurred during saveHTML: {traceback.format_exc()}"
            return
        except selenium.common.exceptions.WebDriverException:
            self.hasBrowserDied = True
            self.browserDeathReason = f"Following fatal error occurred during saveHTML: {traceback.format_exc()}"
            return
        except urllib3.exceptions.ProtocolError:
            self.hasBrowserDied = True
            self.browserDeathReason = f"Following fatal error occurred during saveHTML: {traceback.format_exc()}"
            return
        except AttributeError:
            self.hasBrowserDied = True
            self.browserDeathReason = f"Following fatal error occurred during saveHTML: {traceback.format_exc()}"
            return
