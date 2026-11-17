import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from demo_app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://localhost:{port}"
    print("=" * 60)
    print(" ⚖️  JUDGEBERT INTERACTIVE DEMO APPLICATION")
    print(f" URL: {url}")
    print("=" * 60)
    print("Serving on port", port)
    app.run(host="0.0.0.0", port=port, debug=False)
