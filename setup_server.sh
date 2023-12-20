#!/bin/bash

# activate craco environment
# conda activate dash3.8

# make sure the redis server is running
if ! redis-cli ping > /dev/null 2>&1; then
  redis-server --daemonize yes --bind 127.0.0.1
  redis-cli ping > /dev/null 2>&1  # the script halts if redis is not now running (failed to start)
fi

mkdir -p Log
celery -A app.celery_app worker --loglevel=INFO >> Log/celery_info.log 2>&1 &
gunicorn --workers=12 --name=craco --bind=127.0.0.1:8024 app:server >> Log/gunicorn.log 2>&1