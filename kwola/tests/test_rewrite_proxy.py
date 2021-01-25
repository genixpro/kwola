
import unittest
from ..components.proxy.RewriteProxy import RewriteProxy
from ..components.plugins.base.ProxyPluginBase import ProxyPluginBase


class TestRewriteProxy(unittest.TestCase):
    def test_canonicalize_url(self):
        self.assertEqual(
            RewriteProxy.canonicalizeUrl("http://kros1.kwola.io/components/navbar/navbar.html"),
            "http://kros1.kwola.io/components/navbar/navbar.html"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://gwl-demo.purewealth.cloud/Content/js/optimized/Login-default--en_ca-gwl-demo-29B8FA5E1203D3BFAEBA04B6EC29D1949F3D63F4.js"),
            "https://gwl-demo.purewealth.cloud/Content/js/optimized/Login-default--en_ca-gwl-demo-__HEXID__.js"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://ajaxgeo.cartrawler.com/webapp-abe-latest/chunks/89ac5efe15d27662d66e.486.chunk.js"),
            "https://ajaxgeo.cartrawler.com/webapp-abe-latest/chunks/__HEXID__.486.chunk.js"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://www.edreams.es/travel/static-content/js/desktop.dp-optimized.runtime.93b54e516eb9de0cbf55.js"),
            "https://www.edreams.es/travel/static-content/js/desktop.dp-optimized.runtime.__HEXID__.js"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://www.edreams.es/travel/static-content/js/desktop.secure.bundle~01d99f6b.2c21596677509e0b5569.js"),
            "https://www.edreams.es/travel/static-content/js/desktop.secure.bundle~01d99f6b.__HEXID__.js"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://connect.facebook.net/signals/config/889397124432025?v=2.9.30&r=stable"),
            "https://connect.facebook.net/signals/config/__LONG__?v=2.9.30&r=stable"
        )

        self.assertEqual(
            RewriteProxy.canonicalizeUrl("https://www.edreams.es/travel/setup.js/index.jsp?&preload=true&_=1608420085846"),
            "https://www.edreams.es/travel/setup.js/index.jsp?&preload=true&_=__LONG__"
        )



    def test_get_cleaned_filename(self):
        self.assertEqual(
            ProxyPluginBase.getCleanedURL("http://kros1.kwola.io/components/navbar/navbar.html"),
            "navbar_html"
        )

        self.assertEqual(
            ProxyPluginBase.getCleanedURL("https://gwl-demo.purewealth.cloud/Content/js/optimized/Login-default--en_ca-gwl-demo-29B8FA5E1203D3BFAEBA04B6EC29D1949F3D63F4.js"),
            "Login_default__en_ca_gwl_demo_29B8FA5E1203D3BFAEBA04B6EC29D1949F3D63F4_js"
        )

        self.assertEqual(
            ProxyPluginBase.getCleanedURL("https://ajaxgeo.cartrawler.com/webapp-abe-latest/chunks/89ac5efe15d27662d66e.486.chunk.js"),
            "89ac5efe15d27662d66e_486_chunk_js"
        )

        self.assertEqual(
            ProxyPluginBase.getCleanedURL("https://www.edreams.es/travel/static-content/js/desktop.dp-optimized.runtime.93b54e516eb9de0cbf55.js"),
            "desktop_dp_optimized_runtime_93b54e516eb9de0cbf55_js"
        )
