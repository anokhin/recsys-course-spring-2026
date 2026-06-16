#!/bin/bash
set -e

echo "Waiting for Redis..."
for i in $(seq 1 30); do
    if python3 -c "import redis; r=redis.Redis(host='redis',port=6379); r.ping()" 2>/dev/null; then
        echo "Redis is ready!"
        break
    fi
    echo "  attempt $i/30..."
    sleep 2
done

exec gunicorn -k gevent -w 2 -b 0.0.0.0:5001 --timeout 180 botify.server:app
