import os
import scanpy as sc
import h5py
import scipy as sp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}

class IterativeSVDImputator(object):
    def __init__(self, svd_params=default_svd_params, iters=2):
        self.missing_values = 0.0
        self.svd_params = svd_params
        self.iters = iters
        self.svd_decomposers = [None for _ in range(self.iters)]

    def fit(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
            self.svd_decomposers[i].fit(transformed_X)
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]

    def transform(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]
        return transformed_X


class Loader(object):
    """ Data loader """

    def __init__(self, config, dataset_name, drop_last=True, kwargs={}):
        """Pytorch data loader

        Args:
            config (dict): Dictionary containing options and arguments.
            dataset_name (str): Name of the dataset to load
            drop_last (bool): True in training mode, False in evaluation.
            kwargs (dict): Dictionary for additional parameters if needed

        """
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(
            dataset_name)
        self.data_max = train_dataset.data_max
        self.data_min = train_dataset.data_min

        # Set the loader for training set
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size*5, shuffle=False, drop_last=False, **kwargs)

    def get_dataset(self, dataset_name):
        """Returns training, validation, and test datasets"""
        # Create dictionary for loading functions of datasets.
        # If you add a new dataset, add its corresponding dataset class here in the form 'dataset_name': ClassName
        loader_map = {'default_loader': scRNADataset}
        # Get dataset. Check if the dataset has a custom class.
        # If not, then assume a tabular data with labels in the first column
        dataset = loader_map[dataset_name] if dataset_name in loader_map.keys(
        ) else loader_map['default_loader']
        # Training  datasets
        train_dataset = dataset(
            self.config, dataset_name=dataset_name, mode='train')
        # Test dataset
        test_dataset = dataset(
            self.config, dataset_name=dataset_name, mode='test')

        # Return
        return train_dataset, test_dataset


class scRNADataset(Dataset):
    def __init__(self, config, dataset_name, mode='train'):
        """Dataset class for scRNA data format.

        Args:
            config (dict): Dictionary containing options and arguments.
            dataset_name (str): Name of the dataset to load
            mode (bool): Defines whether the data is for Train or Test mode

        """

        self.config = config
        if mode == 'train':
            self.iterator = self.prepare_training_pairs
        else:
            self.iterator = self.prepare_test_pairs
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], dataset_name)
        self.data, self.labels = self._load_data()
        self.data_dim = self.data.shape[1]

    def __len__(self):
        """Returns number of samples in the data"""
        return len(self.data)

    def prepare_training_pairs(self, idx):
        sample = self.data[idx]
        sample_tensor = torch.Tensor(sample)
        cluster = int(self.labels[idx])
        return sample, cluster

    def prepare_test_pairs(self, idx):
        sample = self.data[idx]
        cluster = int(self.labels[idx])
        return sample, cluster

    def __getitem__(self, index):
        """Returns batch"""
        return self.iterator(index)

    def _load_data(self):
        """Loads one of many available datasets, and returns features and labels"""

        data, labels = self.load_data(self.data_path)

        n_classes = len(list(set(labels.reshape(-1, ).tolist())))
        self.config["feat_dim"] = data.shape[1]
        if self.config["n_classes"] != n_classes:
            self.config["n_classes"] = n_classes
            print(f"{50 * '>'} Number of classes changed "
                  f"from {self.config['n_classes']} to {n_classes} {50 * '<'}")
        self.data_max = np.max(np.abs(data))
        self.data_min = np.min(np.abs(data))

        return data, labels

    def load_data(self, path):
        """Loads scRNA-seq dataset"""
        data_mat = h5py.File(
            f"{path}.h5", "r")
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])

        if Y.dtype != "int64":
            encoder_x = LabelEncoder()
            Y = encoder_x.fit_transform(Y)

        nb_genes = 1000

        X = np.ceil(X).astype(np.int)
        count_X = X
        print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
        adata = sc.AnnData(X)

        adata = self.normalize(adata,
                               copy=True,
                               highly_genes=nb_genes,
                               size_factors=True,
                               normalize_input=True,
                               logtrans_input=True)
        sorted_genes = adata.var_names[np.argsort(adata.var["mean"])]
        adata = adata[:, sorted_genes]
        X = adata.X.astype(np.float32)

        imputator = IterativeSVDImputator(iters=2)
        imputator.fit(X)
        X = imputator.transform(X)

        return X, Y

    def normalize(self, adata, copy=True, highly_genes=None, filter_min_counts=True,
                  size_factors=True, normalize_input=True, logtrans_input=True):
        """
        Normalizes input data and retains only most variable genes
        (indicated by highly_genes parameter)

        Args:
            adata ([type]): [description]
            copy (bool, optional): [description]. Defaults to True.
            highly_genes ([type], optional): [description]. Defaults to None.
            filter_min_counts (bool, optional): [description]. Defaults to True.
            size_factors (bool, optional): [description]. Defaults to True.
            normalize_input (bool, optional): [description]. Defaults to True.
            logtrans_input (bool, optional): [description]. Defaults to True.

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        if isinstance(adata, sc.AnnData):
            if copy:
                adata = adata.copy()
        elif isinstance(adata, str):
            adata = sc.read(adata)
        else:
            raise NotImplementedError
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        assert 'n_count' not in adata.obs, norm_error
        if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
            if sp.sparse.issparse(adata.X):
                assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
            else:
                assert np.all(adata.X.astype(int) == adata.X), norm_error

        if filter_min_counts:
            sc.pp.filter_genes(adata, min_counts=1)  # 3
            sc.pp.filter_cells(adata, min_counts=1)
        if size_factors or normalize_input or logtrans_input:
            adata.raw = adata.copy()
        else:
            adata.raw = adata
        if size_factors:
            sc.pp.normalize_per_cell(adata)
            adata.obs['size_factors'] = adata.obs.n_counts / \
                np.median(adata.obs.n_counts)
        else:
            adata.obs['size_factors'] = 1.0
        if logtrans_input:
            sc.pp.log1p(adata)
        if highly_genes != None:
            sc.pp.highly_variable_genes(
                adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
        if normalize_input:
            sc.pp.scale(adata)
        return adata


def apply_noise(X, p=[0.2,0.4]):
    p = torch.tensor(p)
    should_swap = torch.bernoulli(p.to(
        X.device) * torch.ones((X.shape)).to(X.device))
    corrupted_X = torch.where(
        should_swap == 1, X[torch.randperm(X.shape[0])], X)
    masked = (corrupted_X != X).float()
    return corrupted_X, masked
