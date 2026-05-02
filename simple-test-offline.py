import os

# Tell LMCache to use CPU RAM as the offload tier
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"  # 5GB of RAM for KV cache

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    kv_transfer_config=ktc,
    max_model_len=4096,
    gpu_memory_utilization=0.6,
)

shared_prefix = "You are a networking expert with deep knowledge of TCP, UDP, QUIC, and congestion control algorithms including Reno, CUBIC, and BBR. You have published extensively on transport protocol design, flow control mechanisms, and network performance optimization. Your expertise covers both theoretical foundations and practical deployment considerations. Please provide detailed technical answers. "

prompts = [shared_prefix + "Explain TCP congestion control."]
prompts2 = [shared_prefix + "Compare TCP to QUIC."]

import time

# First request
print("\n" + "="*60)
print("REQUEST 1 (cold)")
print("="*60)
t1 = time.time()
out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=50))
print(f"Time: {time.time() - t1:.3f}s",f"Cached tokens: {out[0].num_cached_tokens}")
print(out[0].outputs[0].text)

# Second request
print("\n" + "="*60)
print("REQUEST 2 (should hit cache)")
print("="*60)
t2 = time.time()
out2 = llm.generate(prompts2, SamplingParams(temperature=0, max_tokens=50))
print(f"Time: {time.time() - t2:.3f}s",f"Cached tokens: {out2[0].num_cached_tokens}")
print(out2[0].outputs[0].text)