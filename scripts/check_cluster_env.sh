#!/usr/bin/env bash
# Quick environment checks before running multi-node DeepEP benchmarks.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "=== DeepEP cluster environment check ==="
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo

command -v nvidia-smi >/dev/null && pass "nvidia-smi found" || fail "nvidia-smi not found"
command -v python >/dev/null && pass "python found: $(python --version 2>&1)" || fail "python not found"
command -v ibstat >/dev/null && pass "ibstat found" || warn "ibstat not found (RDMA diagnostics unavailable)"

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
pass "GPU count: ${gpu_count}"

python - <<'PY' || fail "deep_ep import failed"
import deep_ep
print(f"deep_ep version: {getattr(deep_ep, '__version__', 'unknown')}")
PY

python - <<'PY' || warn "NCCL python package not found (may still work via system NCCL)"
try:
    import torch
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
except Exception as e:
    raise SystemExit(str(e))
PY

if command -v ibstat >/dev/null; then
    nic="${EP_NIC_NAME:-mlx5_0}"
    if ibstat "${nic}" >/dev/null 2>&1; then
        pass "RDMA NIC ${nic} is present"
        ibstat "${nic}" | grep -E "State:|Rate:" | sed 's/^/  /'
    else
        warn "RDMA NIC ${nic} not found; set EP_NIC_NAME if needed"
        ibstat -l 2>/dev/null | sed 's/^/  available: /' || true
    fi
fi

echo
echo "=== Recommended NCCL / DeepEP env (adjust for your cluster) ==="
cat <<'ENV'
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# export NCCL_SOCKET_IFNAME=eth0      # TCP fallback / bootstrap
# export NCCL_IB_HCA=mlx5_0,mlx5_1      # RDMA devices
# export EP_NIC_NAME=mlx5_0           # DeepEP NIC query target
ENV

echo
pass "Local checks finished"
