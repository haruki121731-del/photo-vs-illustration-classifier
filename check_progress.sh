#!/bin/bash
# 進捗確認スクリプト

echo "=============================================="
echo "  TRAINING PROGRESS CHECK"
echo "=============================================="
echo ""

# プロセス確認
PID=$(cat full_pipeline_run/pipeline.pid 2>/dev/null || echo "N/A")
echo "Pipeline PID: $PID"

if [ "$PID" != "N/A" ] && ps -p "$PID" > /dev/null 2>&1; then
    echo "Status: 🟢 RUNNING"
    echo "CPU/Memory: $(ps aux | grep "$PID" | grep -v grep | awk '{print $3 "% CPU, " $4 "% MEM"}')"
else
    echo "Status: 🔴 STOPPED"
fi

echo ""
echo "=== Recent Logs ==="
tail -30 full_pipeline_run/pipeline.log 2>/dev/null | grep "\[202" || echo "No logs yet"

echo ""
echo "=== Dataset Status ==="
if [ -d "data/massive_raw" ]; then
    ILLUST=$(find data/massive_raw/illustrations -type f 2>/dev/null | wc -l)
    PHOTO=$(find data/massive_raw/photos -type f 2>/dev/null | wc -l)
    echo "Illustrations: $ILLUST images"
    echo "Photos: $PHOTO images"
    echo "Total: $((ILLUST + PHOTO)) images"
fi

echo ""
echo "=== Training Status ==="
if [ -f "full_pipeline_run/training/training_history.json" ]; then
    echo "Training in progress..."
    cat full_pipeline_run/training/training_history.json | tail -5
fi

echo ""
echo "=============================================="
echo "Next check: ./check_progress.sh"
echo "Live log: tail -f full_pipeline_run/pipeline.log"
echo "=============================================="
