#!/usr/bin/env bash

export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-memory-utilization 0.6 \
  --port 8000 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1Dynamic",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_connector_v1"
  }'
# vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#   --gpu-memory-utilization 0.6 \
#   --port 8000 \
#   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'