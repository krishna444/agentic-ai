#!/bin/bash
FIRST_ARGUMENT="${1:---no-build}"
if [[ "$FIRST_ARGUMENT" == "--help" ]]; then
    echo "Deploy.sh [--build/--help/down]"
elif [[ "$FIRST_ARGUMENT" == "--build" ]]; then
    docker compose up -d --build
elif [[ "$FIRST_ARGUMENT" == "--no-build" ]]; then
    docker compose up -d 
elif [[ "$FIRST_ARGUMENT" == "down" ]]; then
    docker compose down
fi