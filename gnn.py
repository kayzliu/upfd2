import pygod
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score

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


def text_encoder(dataset, tokenizer, model, max_content_len=500, max_edges=30, batch_size=5):
    for data, text in enumerate(dataset):
        text = text[:max_edges + 1]
        data.edge_index = data.edge_index[:, :max_edges]
        queries = [get_instruct(i, r, max_content_len) for i, r in enumerate(text)]
        dataloader = DataLoader(queries, batch_size=batch_size)
        emb = []
        for batch in dataloader:
            batch_dict = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt")

            batch_dict.to(model.device)
            outputs = model(**batch_dict)
            x = last_token_pool(outputs.last_hidden_state,
                             batch_dict['attention_mask']).detach().cpu()
            emb.append(x)
        data.x = torch.cat(emb, dim=0)


def run_gnn(path, name, emb_model="Qwen/Qwen3-Embedding-8B",
            gnn='SAGE',
            no_graph=False,
            no_user=False,
            max_content_len=500,
            max_edges=30,
            gpu=0):
    tokenizer = AutoTokenizer.from_pretrained(emb_model, padding_side='left')
    model = AutoModel.from_pretrained(emb_model, device_map='auto')
    model.eval()

    train_set = UPFD2(path, name, 'train')
    val_set = UPFD2(path, name, 'val')
    test_set = UPFD2(path, name, 'test')

    print("Encoding text data...")
    for dataset in [train_set, val_set, test_set]:
        text_encoder(dataset, tokenizer, model, max_content_len, max_edges)

    del tokenizer
    del model

    device = pygod.utils.validate_device(gpu)

    gnn = Net(gnn, 4096, 128, 1).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(60):
        gnn.train()

        total_loss = 0
        label, pred = [], []
        for data, text in train_set:
            data = data.to(device)
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index)
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
        for data, text in val_set:
            data = data.to(device)
            out = gnn(data.x, data.edge_index)
            label.append(data.y.item())
            pred.append((out >= 0.5).long().item())
        val_acc = accuracy_score(label, pred)
        val_f1 = f1_score(label, pred)

        label, pred = [], []
        for data, text in test_set:
            data = data.to(device)
            out = gnn(data.x, data.edge_index)
            label.append(data.y.item())
            pred.append((out >= 0.5).long().item())
        test_acc = accuracy_score(label, pred)
        test_f1 = f1_score(label, pred)

        print(f'Epoch: {epoch:02d} | Train: loss: {loss:.4f}, '
              f'acc: {train_acc:.4f}, f1: {train_f1:.4f} | '
              f'Val: acc: {val_acc:.4f}, f1: {val_f1:.4f} | '
              f'Test: acc: {test_acc:.4f}, f1: {test_f1:.4f}')

    return {
        "acc": test_acc,
        "f1": test_f1,
    }