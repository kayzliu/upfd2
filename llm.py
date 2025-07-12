import os

from tqdm import tqdm
from openai import OpenAI


system_prompt = """
You are a fake news detection assistant analyzing news propagation graph on social networks.
                 
You will be provided with :
- the content of a news (corresponding to the root node in the propagation graph).
- user posts (each corresponding to a subsequent node in the propagation graph).
- the structure of the propagation graph. The edges indicate the propagation relationships.

Based on the content and graph structure, your task is to determine whether the news is ‘Real’ or ‘Fake’.

Output: respond with only the fakes news classification label: 'Real' or 'Fake'.
"""


def create_graph_prompt(data,
                        text,
                        max_content_len=2000,
                        max_edges=100,
                        no_graph=False,
                        no_user=False):
    prompt = "Input:\n"

    for i, content in enumerate(text):
        node_type = "NEWS" if i == 0 else "USER POST"
        prompt += f"Node {i} ({node_type}): "
        if len(content) > max_content_len:
            prompt += f"{content[:max_content_len]}...\n"
        else:
            prompt += f"{content}\n"
        if no_user:
            prompt += "No user post content and propagation graph provided.\n"
            return prompt

    prompt += "\nGraph Structure:\n"

    if no_graph:
        prompt += "No graph structure provided.\n"
    else:
        if data.num_edges > max_edges:
            src_nodes = data.edge_index[0][:max_edges]
            dst_nodes = data.edge_index[1][:max_edges]
        else:
            src_nodes = data.edge_index[0]
            dst_nodes = data.edge_index[1]
        for src, dst in zip(src_nodes, dst_nodes):
            prompt += f"Node {src} propagate to Node {dst},\n"

        if data.num_edges > max_edges:
            prompt += f"and {data.num_edges - max_edges} more edges.\n"

    return prompt


def predict(dataset, model="gpt-4.1-mini", no_graph=False, no_user=False):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    results = []
    for data, text in tqdm(zip(dataset, dataset.text)):
        graph_prompt = create_graph_prompt(data, text, no_graph=no_graph, no_user=no_user)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": graph_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            label_pred = response.choices[0].message.content.strip()
            binary_pred = 1 if label_pred.lower() == 'fake' else 0
        except Exception as e:
            print(f"Error with model {model}: {e}")
            binary_pred = -1

        results.append(binary_pred)

    return results
