import pygod
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected
from vllm import LLM

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


def get_instruct(node_type, query):
    if node_type == 0:
        task = "Given a news, generate a semantic embedding for fake news detection."
    else:
        task = "Given a user post, generate a semantic embedding for fake news detection."
    return f'Instruct: {task}\nQuery:{query}'


def run_gnn(path, name, emb_model="Qwen/Qwen3-Embedding-8B", gnn='SAGE', no_graph=False, no_user=False, gpu=0):
    llm = LLM(model=emb_model, task="embed")

    train_set = UPFD2(path, name, 'train', ToUndirected())
    val_set = UPFD2(path, name, 'val', ToUndirected())
    test_set = UPFD2(path, name, 'test', ToUndirected())

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    device = pygod.utils.validate_device(gpu)

    gnn = Net(gnn, train_set.num_features, 128,
                train_set.num_classes, concat=True).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001,
                                 weight_decay=0.01)

    for epoch in range(60):
        gnn.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            queries = [get_instruct(i, r) for i, r in enumerate(g.text) for g in data]
            outputs = llm.embed(queries)
            data.x = torch.tensor([o.outputs.embedding for o in outputs])

            out = gnn(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs

        loss = total_loss / len(train_loader.dataset)

        gnn.eval()

        total_correct = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            pred = gnn(data.x, data.edge_index, data.batch).argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
            total_examples += data.num_graphs
        train_acc = total_correct / total_examples

        total_correct = total_examples = 0
        for data in val_loader:
            data = data.to(device)
            pred = gnn(data.x, data.edge_index, data.batch).argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
            total_examples += data.num_graphs
        val_acc = total_correct / total_examples

        total_correct = total_examples = 0
        for data in test_loader:
            data = data.to(device)
            pred = gnn(data.x, data.edge_index, data.batch).argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
            total_examples += data.num_graphs
        test_acc = total_correct / total_examples

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
