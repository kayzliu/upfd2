# Qwen3-8B
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4 --api-key token-abc123 --rope-scaling '{"rope_type":"yarn", "factor":2.0, "original_max_position_embeddings":32768}' --max-model-length 65536
# Llama 3.1 8B Instruct
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --api-key token-abc123 --rope-scaling '{"rope_type":"yarn", "factor":2.0, "original_max_position_embeddings":32768}' --max-model-length 65536