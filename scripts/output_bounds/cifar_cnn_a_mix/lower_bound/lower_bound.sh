#!/bin/bash

# Exit immediately if any command fails
set -e

# Get the directory where the script is located (lower_bound/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define project root (4 levels up from lower_bound/)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Load environment variables from .env file in project root
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
else
    echo "Warning: .env file not found at ${PROJECT_ROOT}/.env"
fi


echo "Script directory: ${SCRIPT_DIR}"
echo "Current working directory: $(pwd)"
echo "Project root: ${PROJECT_ROOT}"

# Define paths
# Path to main.py (navigate up 4 levels to project root, then into src/)
MAIN_PY="${SCRIPT_DIR}/../../../../src/main.py"
# Config directory is the current directory
CONFIG_DIR="${SCRIPT_DIR}"

# Check if main.py exists
if [ ! -f "${MAIN_PY}" ]; then
    echo "Error: ${MAIN_PY} not found. Please check the path to main.py."
    exit 1
fi

# Check if config directory exists
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "Error: Config directory ${CONFIG_DIR} not found."
    exit 1
fi

# Check if lower_bound.yaml exists
if [ ! -f "${CONFIG_DIR}/lower_bound.yaml" ]; then
    echo "Error: ${CONFIG_DIR}/lower_bound.yaml not found."
    exit 1
fi

# Debug environment variables
echo "DATA_DIR: ${DATA_DIR}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"


python "${MAIN_PY}" \
    --config-path "${CONFIG_DIR}" \
    --config-name lower_bound.yaml