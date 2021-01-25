
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import KwolaCoreConfiguration
import shutil
import traceback
from ..config.logger import getLogger, setupLocalLogging
from ..components.utils.deunique import deuniqueString


class TestDeuniqueString(unittest.TestCase):
    def test_deunique_strings(self):
        self.assertEqual(
            deuniqueString("https://gwl-demo.purewealth.cloud/Content/js/optimized/Login-default--en_ca-gwl-demo-29B8FA5E1203D3BFAEBA04B6EC29D1949F3D63F4.js", deuniqueMode="url"),
            "https://gwl-demo.purewealth.cloud/Content/js/optimized/Login-default--en_ca-gwl-demo-.js"
        )
        self.assertEqual(
            deuniqueString("https://gwl-demo.purewealth.cloud/Content/js/optimized/site-extjs40-default--en_ca-gwl-demo-7439FD767509509BE412B5B9B7074340B1331B8F.js", deuniqueMode="url"),
            "https://gwl-demo.purewealth.cloud/Content/js/optimized/site-extjs40-default--en_ca-gwl-demo-.js"
        )
