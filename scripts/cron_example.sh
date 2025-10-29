# file: projects/sp500-rank-tracker/scripts/cron_example.sh
#!/usr/bin/env bash
# why: cron entry example
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sp500_tracker.cli fetch --dropped-rank 9999
