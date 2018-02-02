from config import NB_CLASSES, DIM
import numpy as np
import random


class CAT:
    def __init__(self, items, popularity):
        self.available_item_ids = list(items)
        self.asked = []
        self.fisher_history = np.zeros((DIM, DIM))
        self.popularity = popularity

    def update_probas(self, cdf, pdf, item_bias, item_features):
        nb = len(item_bias)
        self.cdf = np.column_stack((np.ones(nb), cdf, np.zeros(nb)))
        self.cdfp = self.cdf * (1 - self.cdf)
        self.cdfp_delta = self.cdf[:, :-1] - self.cdf[:, 1:]
        self.pdf = pdf
        assert self.cdfp_delta.shape == self.pdf.shape
        self.item_bias = item_bias
        self.item_features = item_features


class Random(CAT):
    def next_item(self):
        chosen_item_id = random.choice(self.available_item_ids)
        self.asked.append(chosen_item_id)
        self.available_item_ids.remove(chosen_item_id)
        return chosen_item_id


class Popular(CAT):
    def next_item(self):
        best_pos = self.popularity[self.available_item_ids].argmax()
        self.asked.append(self.available_item_ids[best_pos])
        return self.available_item_ids.pop(best_pos)


class Fisher(CAT):
    def next_item(self):
        kept = self.available_item_ids
        self.eigenvalues = (self.cdfp_delta[kept, :] ** 2 / self.pdf[kept, :]).sum(axis=1)
        V = self.item_features[kept, :]
        contestants = np.einsum('li,ij,ki->ijk', np.diag(self.eigenvalues), V, V.T)
        self.scores = np.array([np.linalg.det(self.fisher_history + contestant) for contestant in contestants])
        best_pos = self.scores.argmax()
        self.fisher_history += contestants[best_pos]
        self.asked.append(self.available_item_ids[best_pos])
        return self.available_item_ids.pop(best_pos)
