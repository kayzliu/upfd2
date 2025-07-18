import pygod
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score

from dataset import UPFD2


class Net(torch.nn.Module):
    def __init__(self, gnn, in_channels, hidden_channels, out_channels,
                 concat=False):
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

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)


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


def run_gnn(path, name, emb_model="Qwen/Qwen3-Embedding-8B",
            gnn='SAGE',
            no_graph=False,
            no_user=False,
            max_content_len=500,
            max_edges=30,
            gpu=0):
    tokenizer = AutoTokenizer.from_pretrained(emb_model, padding_side='left')
    model = AutoModel.from_pretrained(emb_model, device_map='auto')

    train_set = UPFD2(path, name, 'train')
    val_set = UPFD2(path, name, 'val')
    test_set = UPFD2(path, name, 'test')

    device = pygod.utils.validate_device(gpu)

    gnn = Net(gnn, 4096, 128, 2, concat=True).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001,
                                 weight_decay=0.01)

    for epoch in range(60):
        gnn.train()

        total_loss = 0
        total_correct = 0
        for data, text in train_set:
            data = data.to(device)
            optimizer.zero_grad()

            queries = [get_instruct(i, r, max_content_len) for i, r in enumerate(text)]
            if len(queries) > max_edges:
                queries = queries[:max_edges]
                data.edge_index = data.edge_index[:, :max_edges]

            batch_dict = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )

            batch_dict.to(model.device)
            outputs = model(**batch_dict)
            data.x = last_token_pool(outputs.last_hidden_state,
                                         batch_dict['attention_mask'])

            out = gnn(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = out.argmax(dim=-1)
            total_correct += int((pred == data.y).sum())

        loss = total_loss / len(train_set)
        train_acc = total_correct / len(train_set)

        total_correct = 0
        for data in val_set:
            data = data.to(device)
            pred = gnn(data.x, data.edge_index, data.batch).argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
        val_acc = total_correct / len(val_set)

        total_correct = 0
        for data in test_set:
            data = data.to(device)
            pred = gnn(data.x, data.edge_index, data.batch).argmax(dim=-1)
            label = data.y.cpu().numpy()
            total_correct += int((pred == data.y).sum())
        test_acc = total_correct / len(test_set)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    return {
        "acc": accuracy_score(label, pred),
        "f1": f1_score(label, pred),
    }