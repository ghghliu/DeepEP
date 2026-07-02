#!/usr/bin/env bash
# Launch DeepEP V2 multi-node benchmark (tests/elastic/test_ep.py) across SSH hosts.
#
# Prerequisites:
#   - Passwordless SSH from the launch node to every host in the hostfile
#   - DeepEP installed and importable on every node (same path recommended)
#   - Same GPU count per node (or set slots= per line in hostfile)
#
# Usage:
#   ./scripts/run_multinode_benchmark.sh --hostfile scripts/hostfile.example
#   ./scripts/run_multinode_benchmark.sh --hostfile hosts.txt --deepep-dir /path/to/deepep -- --test-first-only --skip-check
#
# Hostfile format (one node per line):
#   <ssh_host> [slots=<num_gpus>]
# The first host is the master (RANK=0, MASTER_ADDR).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOSTFILE=""
DEEPEP_DIR="${DEEPEP_DIR:-${REPO_ROOT}}"
SSH_USER="${SSH_USER:-}"
MASTER_PORT="${MASTER_PORT:-8361}"
PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-/tmp/deepep_multinode_logs}"
EXTRA_TEST_ARGS=()

usage() {
    cat <<'EOF'
Usage: run_multinode_benchmark.sh --hostfile <file> [options] [-- test_ep.py args...]

Options:
  --hostfile <file>     Required. One SSH host per line; first line is master.
  --deepep-dir <path>   DeepEP repo root on remote nodes (default: this repo)
  --ssh-user <user>     SSH username (default: current user)
  --master-port <port>  MASTER_PORT for torch.distributed (default: 8361)
  --python <bin>        Python executable on remote nodes (default: python)
  --log-dir <path>      Directory for per-node logs on each node (default: /tmp/deepep_multinode_logs)
  -h, --help            Show this help

Environment (optional, forwarded to all nodes):
  NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_DISABLE, NCCL_NET_GDR_LEVEL,
  EP_NIC_NAME, EP_OVERRIDE_RDMA_SL, CUDA_VISIBLE_DEVICES

Examples:
  ./scripts/run_multinode_benchmark.sh --hostfile scripts/hostfile.example
  ./scripts/run_multinode_benchmark.sh --hostfile hosts.txt -- --test-first-only --skip-perf-test
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile)
            HOSTFILE="$2"; shift 2 ;;
        --deepep-dir)
            DEEPEP_DIR="$2"; shift 2 ;;
        --ssh-user)
            SSH_USER="$2"; shift 2 ;;
        --master-port)
            MASTER_PORT="$2"; shift 2 ;;
        --python)
            PYTHON="$2"; shift 2 ;;
        --log-dir)
            LOG_DIR="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift
            EXTRA_TEST_ARGS=("$@")
            break ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ -z "${HOSTFILE}" ]]; then
    echo "Error: --hostfile is required" >&2
    usage
    exit 1
fi
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "Error: hostfile not found: ${HOSTFILE}" >&2
    exit 1
fi

ssh_target() {
    local host="$1"
    if [[ -n "${SSH_USER}" ]]; then
        echo "${SSH_USER}@${host}"
    else
        echo "${host}"
    fi
}

# Parse hostfile
declare -a HOSTS=()
declare -a SLOTS=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="$(echo "${line}" | xargs)"
    [[ -z "${line}" ]] && continue
    host="${line%% *}"
    slots=8
    if [[ "${line}" == *slots=* ]]; then
        slots="${line##*slots=}"
        slots="${slots%% *}"
    fi
    HOSTS+=("${host}")
    SLOTS+=("${slots}")
done < "${HOSTFILE}"

NUM_NODES="${#HOSTS[@]}"
if [[ "${NUM_NODES}" -lt 1 ]]; then
    echo "Error: hostfile contains no hosts" >&2
    exit 1
fi

MASTER_HOST="${HOSTS[0]}"
MASTER_SSH="$(ssh_target "${MASTER_HOST}")"

MASTER_IP="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${MASTER_SSH}" \
    "hostname -I 2>/dev/null | awk '{print \$1}'" 2>/dev/null || true)"
if [[ -z "${MASTER_IP}" ]]; then
    MASTER_IP="${MASTER_HOST}"
fi

echo "=== DeepEP multi-node benchmark ==="
echo "Nodes: ${NUM_NODES}"
echo "Master: ${MASTER_SSH} (${MASTER_IP}:${MASTER_PORT})"
echo "DeepEP dir: ${DEEPEP_DIR}"
echo "Remote logs: ${LOG_DIR}/rank<N>_<host>.log on each node"
if ((${#EXTRA_TEST_ARGS[@]})); then
    echo "Test args: ${EXTRA_TEST_ARGS[*]}"
else
    echo "Test args: <default>"
fi
echo

PIDS=()
declare -a TARGETS=()
declare -a LOG_FILES=()

for rank in "${!HOSTS[@]}"; do
    host="${HOSTS[$rank]}"
    target="$(ssh_target "${host}")"
    slots="${SLOTS[$rank]}"
    log_file="${LOG_DIR}/rank${rank}_${host}.log"

    TARGETS+=("${target}")
    LOG_FILES+=("${log_file}")

    test_args_str=""
    if ((${#EXTRA_TEST_ARGS[@]})); then
        test_args_str="${EXTRA_TEST_ARGS[*]}"
    fi

    echo "Launching RANK=${rank} on ${target} (${slots} GPUs) -> ${log_file}"

    ssh -o BatchMode=yes -o ConnectTimeout=15 "${target}" bash -s -- \
        "${DEEPEP_DIR}" "${MASTER_IP}" "${MASTER_PORT}" "${NUM_NODES}" "${rank}" \
        "${PYTHON}" "${slots}" "${log_file}" "${test_args_str}" <<'REMOTE' &
set -euo pipefail
DEEPEP_DIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"
WORLD_SIZE="$4"
RANK="$5"
PYTHON="$6"
NUM_PROCS="$7"
LOG_FILE="$8"
TEST_ARGS="$9"

cd "${DEEPEP_DIR}"
export MASTER_ADDR MASTER_PORT WORLD_SIZE RANK
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
mkdir -p "$(dirname "${LOG_FILE}")"

# shellcheck disable=SC2086
${PYTHON} tests/elastic/test_ep.py --num-processes "${NUM_PROCS}" ${TEST_ARGS} \
    > "${LOG_FILE}" 2>&1
REMOTE
    PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done

echo
echo "=== Remote log tails ==="
for idx in "${!TARGETS[@]}"; do
    target="${TARGETS[$idx]}"
    log_file="${LOG_FILES[$idx]}"
    echo "--- ${target}:${log_file} ---"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${target}" "tail -n 40 '${log_file}'" 2>/dev/null \
        || echo "(failed to read remote log)"
    echo
done

if [[ "${FAIL}" -ne 0 ]]; then
    echo "Multi-node benchmark failed. Inspect remote logs listed above." >&2
    exit 1
fi

echo "Multi-node benchmark finished successfully."
