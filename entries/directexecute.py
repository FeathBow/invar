import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.direct import main

if __name__ == "__main__":
    main(execution_only=True)
