import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
# The eval CLIs live under scripts/eval/ (split out of src/asa in 65da070), so
# put that directory on the path to import them by module name in tests.
sys.path.insert(0, str(Path(__file__).parent / "scripts" / "eval"))
