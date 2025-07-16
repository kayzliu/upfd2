# politifact
python exp.py --path ./data --name politifact --method zero-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method thinking --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method one-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method two-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method three-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method gnn --llm Qwen/Qwen3-8B

# gossipcop
python exp.py --path ./data --name gossipcop --method zero-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name gossipcop --method thinking --llm Qwen/Qwen3-8B
python exp.py --path ./data --name gossipcop --method one-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name gossipcop --method two-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name gossipcop --method three-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name gossipcop --method gnn --llm Qwen/Qwen3-8B

# fakeddit
python exp.py --path ./data --name fakeddit --method zero-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name fakeddit --method thinking --llm Qwen/Qwen3-8B
python exp.py --path ./data --name fakeddit --method one-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name fakeddit --method two-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name fakeddit --method three-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name fakeddit --method gnn --llm Qwen/Qwen3-8B

# Data Variants
python exp.py --path ./data --name politifact --method one-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method one-shot --llm Qwen/Qwen3-8B --no-graph
python exp.py --path ./data --name politifact --method one-shot --llm Qwen/Qwen3-8B --no-user

# LLM Variants
python exp.py --path ./data --name politifact --method one-shot --llm Qwen/Qwen3-8B
python exp.py --path ./data --name politifact --method one-shot --llm meta-llama/Llama-3.1-8B-Instruct
python exp.py --path ./data --name politifact --method one-shot --llm gpt-4.1-nano
python exp.py --path ./data --name politifact --method one-shot --llm gpt-4.1-mini
python exp.py --path ./data --name politifact --method one-shot --llm gpt-4.1