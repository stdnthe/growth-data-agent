#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "当前系统不是 macOS，无法使用 Keychain 脚本。"
  exit 1
fi

SERVICE_NAME="${DEEPSEEK_KEYCHAIN_SERVICE:-growth-analysis-agent/DEEPSEEK_API_KEY}"
ACCOUNT_NAME="${DEEPSEEK_KEYCHAIN_ACCOUNT:-$USER}"

KEY_VALUE="${1:-}"
if [[ -z "${KEY_VALUE}" ]]; then
  read -r -s -p "请输入 DeepSeek API Key: " KEY_VALUE
  echo
fi

if [[ -z "${KEY_VALUE}" ]]; then
  echo "DeepSeek API Key 不能为空。"
  exit 1
fi

if ! security add-generic-password \
  -a "${ACCOUNT_NAME}" \
  -s "${SERVICE_NAME}" \
  -w "${KEY_VALUE}" \
  -U >/dev/null 2>&1; then
  echo "写入 Keychain 失败，请确认已允许终端访问钥匙串。"
  exit 1
fi

echo "已保存到 macOS Keychain。"
echo "service=${SERVICE_NAME}"
echo "account=${ACCOUNT_NAME}"
