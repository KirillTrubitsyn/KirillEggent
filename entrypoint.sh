#!/bin/bash
set -e

# Ensure data directories exist (Railway mounts volume at /app/data as root)
mkdir -p /app/data/tmp /app/data/settings /app/data/projects /app/data/chats 2>/dev/null || true

if [ "$(id -u)" = "0" ]; then
  # Running as root — fix ownership then drop to "node" via gosu
  chown -R node:node /app/data 2>/dev/null || true
  exec gosu node "$@"
else
  exec "$@"
fi
