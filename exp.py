import argparse

from llm import run_llm
from gnn import run_gnn


def main(args):

    if args.method == "zero-shot":
        res = run_llm(path=args.path, name=args.name, model=args.model, no_graph=args.no_graph, no_user=args.no_user)
    elif args.method == "thinking":
        res = run_llm(path=args.path, name=args.name, model=args.model, thinking=True, no_graph=args.no_graph, no_user=args.no_user)
    elif args.method == "one-shot":
        res = run_llm(path=args.path, name=args.name, model=args.model, class_examples=1, no_graph=args.no_graph, no_user=args.no_user)
    elif args.method == "two-shot":
        res = run_llm(path=args.path, name=args.name, model=args.model, class_examples=2, no_graph=args.no_graph, no_user=args.no_user)
    elif args.method == "three-shot":
        res = run_llm(path=args.path, name=args.name, model=args.model, class_examples=3, no_graph=args.no_graph, no_user=args.no_user)
    elif args.method == "gnn":
        res = run_gnn(path=args.path, name=args.name, emb_model=args.model, no_graph=args.no_graph, no_user=args.no_user, gpu=args.gpu)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Dataset: {args.name}, Method: {args.method}, "
          f"Accuracy: {res['acc']:.4f}, F1 Score: {res['f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPFD2 experiment")
    parser.add_argument("--path", type=str, default="./data", help="path for the dataset")
    parser.add_argument("--name", type=str, default="politifact", choices=['politifact', 'gossipcop', 'fakeddit'], help="Dataset name")
    parser.add_argument("--method", type=str, default="gnn", choices=['zero-shot', 'thinking', 'one-shot', 'two-shot', 'three-shot', 'gnn'], help="Method for experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="LLM model to use for predictions")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for CUDA, -1 for CPU")
    parser.add_argument("--no-graph", action="store_true", help="Disable graphical output")
    parser.add_argument("--no-user", action="store_true", help="Disable user post content in prompts")
    args = parser.parse_args()

    main(args)
