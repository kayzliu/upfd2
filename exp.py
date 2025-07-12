import argparse

from sklearn.metrics import accuracy_score, f1_score

from dataset import UPFD2
from llm import predict


def main(args):
    test_set = UPFD2(root=args.path, name=args.name, split='test')

    if args.method == "gnn":
        pred = None
    elif args.method == "llm":
        pred = predict(test_set, model=args.model, no_graph=args.no_graph, no_user=args.no_user)
    else:
        raise ValueError("Method must be either 'gnn' or 'llm'.")

    labels = [data.y.item() for data in test_set]

    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)

    print(f"Dataset: {args.name}, Method: {args.method}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPFD2 experiment")
    parser.add_argument("--path", type=str, default="./data", help="path for the dataset")
    parser.add_argument("--name", type=str, default="politifact", choices=['politifact', 'gossipcop', 'fakeddit'], help="Dataset name")
    parser.add_argument("--method", type=str, default="gnn", choices=['gnn', 'llm', ''], help="Method for experiment")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM model to use for predictions")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for CUDA, -1 for CPU")
    parser.add_argument("--no-graph", action="store_true", help="Disable graphical output")
    parser.add_argument("--no-user", action="store_true", help="Disable user post content in prompts")
    args = parser.parse_args()

    main(args)
