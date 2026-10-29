cd /dpc-zhouy/zhouy/hyper-parallel-master
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate hp29
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
source /dpc-zhouy/usr/local/Ascend/cann/set_env.sh
source /dpc-zhouy/usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_CONNECT_TIMEOUT=180
export HCCL_EXEC_TIMEOUT=180

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
LOG_FILE="./logs_sciowl/debug_${TIMESTAMP}.log"

mkdir -p "$(dirname "${LOG_FILE}")"

# export ASCEND_LAUNCH_BLOCKING=1
torchrun --standalone --nproc_per_node=8 scripts/train_lm.py ./scripts/sciowl.yaml 2>&1 | tee ${LOG_FILE}
