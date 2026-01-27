import argparse
import json
from datetime import datetime


def main() -> None:
  parser = argparse.ArgumentParser(description="Freeze mapping bundle")
  parser.add_argument("--mapping_bundle_path", required=True)
  parser.add_argument("--output_path", required=True)
  args = parser.parse_args()

  payload = {
    "mapping_bundle_path": args.mapping_bundle_path,
    "output_path": args.output_path,
    "status": "NOT_IMPLEMENTED",
    "frozen_at": datetime.utcnow().isoformat() + "Z"
  }
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
