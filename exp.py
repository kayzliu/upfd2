import argparse

from llm import predict


def main(args):

    if args.method == "gnn":
        pass
    elif args.method == "zero-shot":
        res = predict(path=args.path, name=args.name, model=args.model)
    elif args.method == "thinking":
        res = predict(path=args.path, name=args.name, model=args.model, thinking=True)
    elif args.method == "one-shot":
        res = predict(path=args.path, name=args.name, model=args.model, class_examples=1)
    elif args.method == "three-shot":
        res = predict(path=args.path, name=args.name, model=args.model, class_examples=3)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Dataset: {args.name}, Method: {args.method}, "
          f"Accuracy: {res['acc']:.4f}, F1 Score: {res['f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPFD2 experiment")
    parser.add_argument("--path", type=str, default="./data", help="path for the dataset")
    parser.add_argument("--name", type=str, default="politifact", choices=['politifact', 'gossipcop', 'fakeddit'], help="Dataset name")
    parser.add_argument("--method", type=str, default="gnn", choices=['zero-shot', 'thinking', 'one-shot', 'three-shot', 'gnn'], help="Method for experiment")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM model to use for predictions")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for CUDA, -1 for CPU")
    parser.add_argument("--no-graph", action="store_true", help="Disable graphical output")
    parser.add_argument("--no-user", action="store_true", help="Disable user post content in prompts")
    args = parser.parse_args()

    main(args)
