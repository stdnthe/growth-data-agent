#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "当前系统不是 macOS，无法使用 Keychain 脚本。"
  exit 1
fi

SERVICE_NAME="${DEEPSEEK_KEYCHAIN_SERVICE:-growth-analysis-agent/DEEPSEEK_API_KEY}"
ACCOUNT_NAME="${DEEPSEEK_KEYCHAIN_ACCOUNT:-$USER}"

if security delete-generic-password -a "${ACCOUNT_NAME}" -s "${SERVICE_NAME}" >/dev/null 2>&1; then
  echo "已删除 Keychain 中的 DeepSeek API Key。"
else
  echo "未找到可删除的 Keychain 项（service=${SERVICE_NAME}, account=${ACCOUNT_NAME}）。"
fi
