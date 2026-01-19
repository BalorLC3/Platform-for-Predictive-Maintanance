import sys
from pathlib import Path

# Add the project root to the Python path.
# This allows pytest to find the 'src' module.
# __file__ is the path to the current file (conftest.py)
# .parent is the 'tests' directory
# .parent again is the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"\n--- Pytest using project root: {project_root} ---")