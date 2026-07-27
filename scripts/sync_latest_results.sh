#!/usr/bin/env bash
# Copy the latest training run's reports and dashboard to assets/ for git tracking.
# Run after training completes:
#   bash scripts/sync_latest_results.sh
set -euo pipefail

LATEST=$(ls -t runs/image_codec/ 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "No training runs found under runs/image_codec/"
  exit 1
fi

REPORTS="runs/image_codec/$LATEST/reports"
DASHBOARD="$REPORTS/dashboard.json"
if [ ! -f "$DASHBOARD" ]; then
  echo "No dashboard.json in $REPORTS — training may not have completed."
  exit 1
fi

echo "Syncing $LATEST to assets/ ..."
mkdir -p assets

# Copy report images
cp "$REPORTS/original.png" "assets/reconstruction_original_$LATEST.png"
cp "$REPORTS/reconstruction.png" "assets/reconstruction_result_$LATEST.png"
cp "$REPORTS/error.png" "assets/reconstruction_error_$LATEST.png"

# Copy bitstream and dashboard
cp "$REPORTS/reconstruction.kky" "assets/reconstruction_$LATEST.kky"
cp "$DASHBOARD" "assets/dashboard_$LATEST.json"

# Remove previous run's assets (keep only the newest)
for f in assets/reconstruction_original_*.png; do
  name=$(basename "$f")
  ts=$(echo "$name" | sed 's/reconstruction_original_//;s/\.png//')
  if [ "$ts" != "$LATEST" ]; then
    rm -f "assets/reconstruction_original_$ts.png" \
          "assets/reconstruction_result_$ts.png" \
          "assets/reconstruction_error_$ts.png" \
          "assets/reconstruction_$ts.kky" \
          "assets/dashboard_$ts.json"
  fi
done

echo "Done. Run: git add assets/ && git commit -m 'sync results $LATEST'"
