import argparse
import json
from datetime import datetime


def main() -> None:
  parser = argparse.ArgumentParser(description="Deterministic pipeline job")
  parser.add_argument("--version_id", required=True)
  parser.add_argument("--mapping_bundle_id")
  parser.add_argument("--config_path", default="shared/config")
  args = parser.parse_args()

  # Placeholder entrypoint to wire the job contract.
  payload = {
    "version_id": args.version_id,
    "mapping_bundle_id": args.mapping_bundle_id,
    "started_at": datetime.utcnow().isoformat() + "Z",
    "status": "NOT_IMPLEMENTED"
  }
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
