# Shared helper: build a CUDA_VISIBLE_DEVICES list of NUM_GPU cards starting at
# START_GPU. Sourced by the training scripts.
build_cuda_visible_devices() {
    local num_gpu=$1
    local start_gpu=${2:-0}
    local list=""
    for ((i = 0; i < num_gpu; i++)); do
        local id=$((start_gpu + i))
        if [ -z "$list" ]; then list="$id"; else list="$list,$id"; fi
    done
    echo "$list"
}
