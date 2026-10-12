#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Compatibility entry point; model creation is orchestrated by create_models.sh.
set -euo pipefail
exec "$(dirname "$0")/../create_models.sh" "$@"
