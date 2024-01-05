import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from typing import Iterable


class DataLoader:
    # for the user features and user-item list
    def __init__(
        self,
        user_features_df: pd.DataFrame,
        user_items_df: pd.DataFrame,
        train_size: float = 0.8,
    ):
        if "USER_ID" not in user_features_df.columns:
            raise ValueError("user_features_df should have a 'USER_ID' column.")
        if "USER_ID" not in user_items_df.columns:
            raise ValueError("user_items_df should have a 'USER_ID' column.")
        if "items" not in user_items_df.columns:
            raise ValueError("user_items_df should have a 'items' column.")

        # only include user-items that have user features
        self.user_items_df = user_items_df[
            user_items_df.USER_ID.isin(user_features_df.USER_ID)
        ]
        self.user_items_df.drop_duplicates(inplace=True)

        self.user_features_df = user_features_df.set_index("USER_ID")

        unique_items = self.user_items_df.items.unique()
        self.item_idx = dict(zip(unique_items, range(unique_items.shape[0])))

        self.n_users = self.user_items_df.USER_ID.nunique()
        self.n_items = unique_items.shape[0]
        self.n_user_features = self.user_features_df.shape[1]

        rand_idx = np.random.permutation(len(self.user_items_df))
        train_size = int(0.8 * len(self.user_items_df))
        self.train_dataset = self.user_items_df.iloc[rand_idx[:train_size]]
        self.val_dataset = self.user_items_df.iloc[rand_idx[train_size:]]
        self.negative_train_df = (
            self.train_dataset.groupby("USER_ID")
            .items.apply(lambda x: set(self.item_idx) - set(x))
            .reset_index()
            .explode("items")
        )
        self.negative_val_df = (
            self.val_dataset.groupby("USER_ID")
            .items.apply(lambda x: set(self.item_idx) - set(x))
            .reset_index()
            .explode("items")
        )

    def train_dataloader(self, batch_size):
        neg2pos_ratio = 4
        neg_size = int(batch_size / (neg2pos_ratio + 1) * neg2pos_ratio)
        df = self.train_dataset.sample(frac=1)
        n = len(df)
        for i in range(0, n, batch_size):
            if i + batch_size <= n:
                yield pd.concat(
                    [
                        df.iloc[i : i + batch_size - neg_size].assign(allocation=1),
                        self.negative_train_df.sample(neg_size).assign(allocation=0),
                    ]
                ).reset_index(drop=True)
            else:
                break

    def valid_dataloader(self, batch_size):
        neg2pos_ratio = 4
        neg_size = int(batch_size / (neg2pos_ratio + 1) * neg2pos_ratio)
        df = self.val_dataset.sample(frac=1)
        n = len(df)
        for i in range(0, n, batch_size):
            if i + batch_size <= n:
                yield pd.concat(
                    [
                        df.iloc[i : i + batch_size - neg_size].assign(allocation=1),
                        self.negative_val_df.sample(neg_size).assign(allocation=0),
                    ]
                ).reset_index(drop=True)
            else:
                break

    def item2longtensor(self, item_list):
        return torch.LongTensor([self.item_idx[f] for f in item_list])


class CFNet(nn.Module):
    def __init__(self, data_loader: DataLoader, n_latent_factors: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader

        super().__init__()
        self.item_embeddings = nn.Embedding(data_loader.n_items, n_latent_factors)
        self.user_net = nn.Linear(data_loader.n_user_features, n_latent_factors)
        self = self.to(torch.float64).to(self.device)

    def forward(self, cst_ids: Iterable, items: Iterable):
        if len(cst_ids) != len(items):
            raise ValueError("cst_ids and items should have the same length.")

        user_features = torch.tensor(
            self.data_loader.user_features_df.loc[cst_ids].values, device=self.device
        )
        user_latent = self.user_net(user_features)

        item_ids = self.data_loader.item2longtensor(items).to(self.device)
        item_latent = self.item_embeddings(item_ids)

        dot_prod = (user_latent * item_latent).sum(axis=1)
        return torch.sigmoid(dot_prod)

    def calculate_item_recommendations(self):
        # perform the matrix multiplication on cpu/ numpy
        user_net = self.user_net.weight.detach().to("cpu").numpy()
        user_embeddings = self.data_loader.user_features_df @ user_net.T
        item_embeddings = self.item_embeddings.weight.detach().to("cpu").numpy()
        similarity_df = 1 / (1 + np.exp(-user_embeddings @ item_embeddings.T))
        similarity_df.columns = self.data_loader.item_idx.keys()
        return similarity_df


class CFNetTrainer:
    def __init__(self, model, data_loader, batch_size, num_epochs, learning_rate=0.001):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.1
        )
        self.criterion = torch.nn.BCELoss()

    def calculate_metrics(self, predictions, targets):
        auc = roc_auc_score(targets, predictions)
        accuracy = accuracy_score(targets, (predictions > 0.5).astype("float"))
        ap = average_precision_score(targets, predictions)
        return auc, accuracy, ap

    def train(self):
        self.model.train()
        self.model.to(self.model.device)
        train_loss_history = []

        total_epochs = self.current_epoch
        for epoch in range(self.num_epochs - total_epochs):
            total_loss = 0
            num_batches = 0

            # Training loop
            for batch_data in self.data_loader.train_dataloader(self.batch_size):
                cst_ids = batch_data["USER_ID"]
                items = batch_data["items"]
                labels = batch_data["allocation"]

                self.optimizer.zero_grad()
                predictions = self.model(cst_ids, items)
                targets = torch.tensor(
                    labels.values, dtype=torch.float64, device=self.model.device
                )

                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            train_loss = total_loss / num_batches
            train_loss_history.append(train_loss)

            # Validation loop
            self.model.eval()
            val_loss = 0
            num_val_batches = 0
            val_predictions = []
            val_targets = []

            for batch_data in self.data_loader.valid_dataloader(self.batch_size):
                cst_ids = batch_data["USER_ID"]
                items = batch_data["items"]
                labels = batch_data["allocation"]

                predictions = self.model(cst_ids, items)
                targets = torch.tensor(
                    labels.values, dtype=torch.float64, device=self.model.device
                )

                val_loss += self.criterion(predictions, targets).item()
                num_val_batches += 1

                val_predictions.append(predictions.cpu().detach().numpy())
                val_targets.append(targets.cpu().detach().numpy())

            val_loss /= num_val_batches
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)

            auc, accuracy, ap = self.calculate_metrics(val_predictions, val_targets)

            # Print epoch results
            clear_output(wait=True)
            self.current_epoch += 1
            print(f"Epoch {self.current_epoch}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            print(f"AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | AP: {ap:.4f}")

        return train_loss_history

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, lr):
        self.learning_rate = lr
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=0.1
        )

    def continue_training(self, num_epochs):
        self.num_epochs += num_epochs
        train_loss_history = self.train()
        return train_loss_history
