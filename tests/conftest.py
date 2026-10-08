import os
import sys

# Engine tests import Hermes's ContextEngine; point HERMES_SRC at a Hermes checkout.
if os.environ.get("HERMES_SRC"):
    sys.path.insert(0, os.environ["HERMES_SRC"])
