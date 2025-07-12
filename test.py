from dataset import UPFD2

dataset = UPFD2('/Users/kayzliu/GNN-FakeNews/data/', 'politifact', split='train')
i=10
print(len(dataset))
print(dataset[i].edge_index.shape)
print(dataset[i].num_edges)
# print(dataset.text[i][0])
# print(sum([len(dataset.text[i]) for i in range(len(dataset.text))]))
# print(dataset.text[0][0])
# print(dataset.x[0][1])
