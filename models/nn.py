import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = user_embeddings.mul(item_embeddings)
        output = self.fc(embeddings)

        return output.squeeze()


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, int(embedding_dim / 2))
        self.fc3 = nn.Linear(int(embedding_dim / 2), 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = torch.cat([user_embeddings, item_embeddings], axis=1)
        output = nn.ReLU()(self.fc1(embeddings))
        output = self.dropout(output)
        output = nn.ReLU()(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)

        return output.squeeze()


class NeuMF(pl.LightningModule):
    def __init__(self, n_users, n_items, embedding_dim, dropout=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
        )

        self.gmf = nn.Linear(embedding_dim, int(embedding_dim / 2))

        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, int(embedding_dim / 2))

        self.fc_final = nn.Linear(embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)

        user_embeddings_gmf = self.user_embedding_gmf(users)
        item_embeddings_gmf = self.item_embedding_gmf(items)
        embeddings_gmf = user_embeddings_gmf.mul(item_embeddings_gmf)

        output_gmf = self.gmf(embeddings_gmf)
        output = nn.ReLU()(self.fc1(embeddings))
        output = self.dropout(output)
        output = nn.ReLU()(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)

        output = torch.cat([output, output_gmf], dim=1)
        output = self.fc_final(output)

        return output.squeeze()


class Recommender(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        users, items, labels = batch
        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        users, items, labels = batch
        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def validation_epoch_end(self, outputs):
        loss = [o["loss"].mul(o["batch_length"]) for o in outputs]
        length = [o["batch_length"] for o in outputs]

        val_loss = sum(loss) / sum(length)
        self.log("Val Loss", val_loss, prog_bar=True)

        scores = [o["logits"] for o in outputs]
        scores = torch.cat(scores).squeeze().numpy()

        scores = scores.reshape(-1, n_items)
        apak = get_apak(scores, test)

        # Save the metric
        self.log("Val Apak", apak, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)


class DNSRecommender(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        idx_users, csr_arr, popdist = batch

        nonzero = torch.nonzero(csr_arr, as_tuple=True)
        popdist[nonzero] = 0
        popdist = np.divide(popdist, popdist.sum(axis=1, keepdims=True))

        users = []
        items = []
        labels = []
        for id_ in np.arange(idx_users.shape[0]):
            pos_items = torch.nonzero(csr_arr[id_], as_tuple=True)[0]
            pos_users = np.repeat(idx_users[id_], pos_items.shape[0])
            pos_labels = np.zeros(pos_items.shape[0])

            users.append(pos_users)
            items.append(pos_items)
            labels.append(pos_labels)

            neg_items = np.random.choice(np.arange(n_items), size=50, p=popdist[id_, :])
            neg_users = np.repeat(idx_users[id_], neg_items.shape[0])
            neg_labels = np.zeros(neg_items.shape[0])

            users.append(neg_users)
            items.append(neg_items)
            labels.append(neg_labels)

        users = torch.from_numpy(np.concatenate(users))
        items = torch.from_numpy(np.concatenate(items))
        labels = torch.from_numpy(np.concatenate(labels))

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        users, items, labels = batch
        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def validation_epoch_end(self, outputs):
        loss = [o["loss"].mul(o["batch_length"]) for o in outputs]
        length = [o["batch_length"] for o in outputs]

        val_loss = sum(loss) / sum(length)
        self.log("Val Loss", val_loss, prog_bar=True)

        scores = [o["logits"] for o in outputs]
        scores = torch.cat(scores).squeeze().numpy()

        scores = scores.reshape(-1, n_items)
        apak = get_apak(scores, test)

        # Save the metric
        self.log("Val Apak", apak, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)


class DNSRecommender(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        idx_users, csr_arr, popdist = batch

        nonzero = torch.nonzero(csr_arr, as_tuple=True)
        popdist[nonzero] = 0
        popdist = np.divide(popdist, popdist.sum(axis=1, keepdims=True))

        users = []
        items = []
        labels = []
        for id_ in np.arange(idx_users.shape[0]):
            pos_items = torch.nonzero(csr_arr[id_], as_tuple=True)[0]
            pos_users = np.repeat(idx_users[id_], pos_items.shape[0])
            pos_labels = np.zeros(pos_items.shape[0])

            users.append(pos_users)
            items.append(pos_items)
            labels.append(pos_labels)

            neg_items = np.random.choice(np.arange(n_items), size=50, p=popdist[id_, :])
            neg_users = np.repeat(idx_users[id_], neg_items.shape[0])
            neg_labels = np.zeros(neg_items.shape[0])

            users.append(neg_users)
            items.append(neg_items)
            labels.append(neg_labels)

        users = torch.from_numpy(np.concatenate(users))
        items = torch.from_numpy(np.concatenate(items))
        labels = torch.from_numpy(np.concatenate(labels))

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        users, items, labels = batch
        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "batch_length": labels.shape[0],
        }

    def validation_epoch_end(self, outputs):
        loss = [o["loss"].mul(o["batch_length"]) for o in outputs]
        length = [o["batch_length"] for o in outputs]

        val_loss = sum(loss) / sum(length)
        self.log("Val Loss", val_loss, prog_bar=True)

        scores = [o["logits"] for o in outputs]
        scores = torch.cat(scores).squeeze().numpy()

        scores = scores.reshape(-1, n_items)
        apak = get_apak(scores, test)

        # Save the metric
        self.log("Val Apak", apak, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)
