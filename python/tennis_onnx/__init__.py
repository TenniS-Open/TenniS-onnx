import logging
logging.basicConfig(level = logging.INFO,format = '[%(asctime)s] [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

from . import exporter
from . import version_checking
try:
    from . import onnx_tools
except ImportError as e:
    import sys
    sys.stderr.write("Can not use onnx_tools with: {}\n".format(e))
