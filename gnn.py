import os

import openai
import pygod
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from vllm import LLM

from dataset import UPFD2


class Net(torch.nn.Module):
    def __init__(self, gnn, in_channels, hidden_channels, out_channels,
                 concat=True):
        super().__init__()
        self.concat = concat

        if gnn == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif gnn == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        elif gnn == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)

        if self.concat:
            self.lin0 = Linear(in_channels, hidden_channels)
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = h.max(dim=-2)[0]

        if self.concat:
            news = x[0]
            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h


def get_instruct(node_type, query, max_content_len=500):
    if node_type == 0:
        task = "Given a news, generate a semantic embedding for fake news detection."
    else:
        task = "Given a user post, generate a semantic embedding for fake news detection."
    if len(query) > max_content_len:
        query = query[:max_content_len] + "..."
    return f'Instruct: {task}\nQuery:{query}'


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_emb(path, name, emb_model="Qwen/Qwen3-Embedding-8B", max_content_len=500, max_edges=30):

    fpath = os.path.join(path, name, f"{emb_model.split('/')[-1]}_{max_content_len}_{max_edges}.pt") if path and name else ""
    if os.path.exists(fpath):
        set_emb = torch.load(fpath)
    else:
        train_set = UPFD2(path, name, 'train')
        val_set = UPFD2(path, name, 'val')
        test_set = UPFD2(path, name, 'test')

        client = openai.OpenAI(base_url="http://localhost:8000/v1",
                               api_key="token-abc123")

        set_emb = []
        for dataset in [train_set, val_set, test_set]:
            data_emb = []
            for data, text in tqdm(dataset):
                text = text[:max_edges + 1]
                queries = [get_instruct(i, r, max_content_len) for i, r in enumerate(text)]
                response = client.embeddings.create(input=queries, model=emb_model)
                data_emb.append(torch.Tensor([o.embedding for o in response.data]))
            set_emb.append(data_emb)
        torch.save(set_emb, fpath)

    return set_emb


def run_gnn(path, name, emb_model="Qwen/Qwen3-Embedding-8B",
            gnn='SAGE',
            no_graph=False,
            no_user=False,
            max_content_len=500,
            max_edges=30,
            gpu=0):

    train_set = UPFD2(path, name, 'train')
    val_set = UPFD2(path, name, 'val')
    test_set = UPFD2(path, name, 'test')

    train_emb, val_emb, test_emb = load_emb(path, name, emb_model, max_content_len, max_edges)

    device = pygod.utils.validate_device(gpu)

    gnn = Net(gnn, 4096, 128, 1).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(60):
        gnn.train()

        total_loss = 0
        label, pred = [], []
        for (data, _), x in zip(train_set, train_emb):
            data = data.to(device)
            x = x.to(device)
            optimizer.zero_grad()
            out = gnn(x, data.edge_index[:, :max_edges])
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            label.append(data.y.item())
            pred.append((out >= 0.5).long().item())
        loss = total_loss / len(train_set)
        train_acc = accuracy_score(label, pred)
        train_f1 = f1_score(label, pred)

        label, pred = [], []
        for (data, _), x in zip(val_set, val_emb):
            data = data.to(device)
            x = x.to(device)
            out = gnn(x, data.edge_index[:, :max_edges])
            label.append(data.y.item())
            pred.append((out >= 0.5).long().item())
        val_acc = accuracy_score(label, pred)
        val_f1 = f1_score(label, pred)

        label, pred = [], []
        for (data, _), x in zip(test_set, test_emb):
            data = data.to(device)
            x = x.to(device)
            out = gnn(x, data.edge_index[:, :max_edges])
            label.append(data.y.item())
            pred.append((out >= 0.5).long().item())
        test_acc = accuracy_score(label, pred)
        test_f1 = f1_score(label, pred)

        print(f'Epoch: {epoch:02d} | Train: loss: {loss:.4f}, '
              f'acc: {train_acc:.4f}, f1: {train_f1:.4f} | '
              f'Val: acc: {val_acc:.4f}, f1: {val_f1:.4f} | '
              f'Test: acc: {test_acc:.4f}, f1: {test_f1:.4f}')

    return {"acc": test_acc, "f1": test_f1,}
