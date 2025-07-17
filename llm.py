import os

from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

from dataset import UPFD2

system_prompt = """
You are a fake news detection assistant analyzing news propagation graph on social networks.

You will be provided with :
- the content of a news (corresponding to the root node in the propagation graph).
- user posts (each corresponding to a subsequent node in the propagation graph).
- the structure of the propagation graph. The edges indicate the propagation relationships.

Based on the content and graph structure, your task is to determine whether the news is ‘Real’ or ‘Fake’.

Output: respond with only the fakes news classification label: 'Real' or 'Fake'.
"""


def graph_encoder(data,
                  text,
                  no_graph=False,
                  no_user=False,
                  max_content_len=500,
                  max_edges=30):
    prompt = "Input:\n"

    for i, content in enumerate(text):
        if i == 0:
            prompt += f"Node 0 (NEWS): {content}\n"
        else:
            prompt += f"Node {i} (USER POST): "
            if len(content) > max_content_len:
                prompt += f"{content[:max_content_len]}...\n"
            else:
                prompt += f"{content}\n"
        if i >= max_edges:
            prompt += f"and {data.num_edges - max_edges} more nodes.\n"
            break
        if no_user:
            prompt += "No user post content and propagation graph provided.\n"
            return prompt

    prompt += "\nGraph Structure:\n"

    if no_graph:
        prompt += "No graph structure provided.\n"
    else:
        src_nodes = data.edge_index[0][:max_edges]
        dst_nodes = data.edge_index[1][:max_edges]

        for src, dst in zip(src_nodes, dst_nodes):
            prompt += f"Node {src} propagate to Node {dst},\n"

        if data.num_edges > max_edges:
            prompt += f"and {data.num_edges - max_edges} more edges.\n"

    return prompt


def example_encoder(dataset, class_examples=0, no_graph=False, no_user=False):

    prompt = "EXAMPLES:\n\n"

    class_cnt = {'Real': 0, 'Fake': 0}
    for data, text in dataset:
        label = "Fake" if data.y.item() else "Real"
        if class_cnt[label] >= class_examples:
            continue
        prompt += graph_encoder(data, text, no_graph, no_user)
        prompt += f"\nOutput: {label}\n\n"
        class_cnt[label] += 1
        if all(cnt >= class_examples for cnt in class_cnt.values()):
            break

    prompt += "END OF EXAMPLES. Classify the following news:\n\n"
    return prompt


def run_llm(path,
            name,
            model="gpt-4.1-nano",
            thinking=False,
            class_examples=0,
            no_graph=False,
            no_user=False):

    # LLM settings
    if model[0:3] == "gpt":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

    extra_body = None
    if model == "Qwen/Qwen3-8B":
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
    else:
        assert not thinking, "Only 'Qwen/Qwen3-8B' supports thinking mode."

    # Few-shot examples from the validation set
    if class_examples > 0:
        val_set = UPFD2(root=path, name=name, split='val')
        example_prompt = example_encoder(val_set, class_examples, no_graph, no_user)
    else:
        example_prompt = ""

    # Prediction on the test set
    label, pred = [], []
    test_set = UPFD2(root=path, name=name, split='test')
    for data, text in tqdm(test_set):
        graph_prompt = graph_encoder(data, text, no_graph, no_user)
        user_prompt = example_prompt + graph_prompt

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                extra_body=extra_body
            )
            output = response.choices[0].message.content
            if "<think>" in output:
                output = output.split("</think>")[-1]
            label_pred = output.strip().lower()
            binary_pred = 1 if label_pred.startswith('fake') else 0
        except Exception as e:
            print(f"Error with model {model}: {e}")
            binary_pred = 0

        label.append(data.y.item())
        pred.append(binary_pred)

    return {
        "acc": accuracy_score(label, pred),
        "f1": f1_score(label, pred),
    }
