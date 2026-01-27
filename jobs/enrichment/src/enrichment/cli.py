import argparse
import json
from datetime import datetime


def main() -> None:
  parser = argparse.ArgumentParser(description="Enrichment job")
  parser.add_argument("--version_id", required=True)
  parser.add_argument("--source", required=True)
  args = parser.parse_args()

  payload = {
    "version_id": args.version_id,
    "source": args.source,
    "status": "NOT_IMPLEMENTED",
    "started_at": datetime.utcnow().isoformat() + "Z"
  }
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
