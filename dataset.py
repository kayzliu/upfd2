import copy
import csv
import json
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.io import fs, read_txt_array
from torch_geometric.utils import coalesce, cumsum


class UPFD2(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in ['politifact', 'gossipcop', 'fakeddit']
        assert split in ['train', 'val', 'test']

        self.root = root
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
            'val_idx.npy', 'test_idx.npy', f'raw_text.csv'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self) -> None:
        text_path = osp.join(self.raw_dir, 'raw_text.csv')
        with open(text_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            text = [r[0] for r in list(reader)]

        edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index = coalesce(edge_index, num_nodes=len(text))

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'edge_index': edge_slice,
            'y': graph_slice
        }

        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            text_list = [text[node_slice[i]:node_slice[i + 1]] for i in idx]

            self.save(data_list, text_list, path)

    @classmethod
    def save(cls, data_list, text_list, path) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        fs.torch_save((data.to_dict(), slices, data.__class__), path)
        with open(path[:-3]+"_text.json", "w", encoding="utf-8") as f:
            json.dump(text_list, f, ensure_ascii=False, indent=2)

    def load(self, path, data_cls=Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        text = json.load(open(path[:-3]+"_text.json", "r", encoding="utf-8"))
        self.text = text

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def get(self, idx: int):

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list = self.len() * [None]
        self._data_list[idx] = copy.copy(data)

        return data, self.text[idx]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)}, name={self.name})'
