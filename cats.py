from config import NB_CLASSES, DIM
import numpy as np
import random


class CAT:
    def __init__(self, items):
        self.bank = list(items)
        self.asked = []
        self.fisher_history = np.zeros((DIM, DIM))

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
        best_pos = random.randint(0, len(self.bank) - 1)
        self.asked.append(self.bank[best_pos])
        return self.bank.pop(best_pos)


# class Fisher(CAT):
#     def next_item(self):
#         arith = np.arange(NB_CLASSES)
#         left = (arith ** 2 * self.pdf).sum(axis=1)
#         right = (arith * self.pdf).sum(axis=1) ** 2
#         contestants = (left - right) * self.item_bias ** 2
#         best_pos = contestants[self.bank].argmax()
#         return self.bank.pop(best_pos)


class Fisher(CAT):
    def next_item(self):
        self.scores = (self.cdfp_delta ** 2 / self.pdf).sum(axis=1) * self.item_features ** 2
        best_pos = self.scores[self.bank].argmax()
        self.asked.append(self.bank[best_pos])
        return self.bank.pop(best_pos)


class FisherM(CAT):
    def next_item(self):
        self.eigenvalues = (self.cdfp_delta ** 2 / self.pdf).sum(axis=1)
        V = self.item_features
        contestants = np.einsum('li,ij,ki->ijk', np.diag(self.eigenvalues), V, V.T)
        self.scores = np.array([np.linalg.det(self.fisher_history + contestant) for contestant in contestants])
        best_pos = self.scores[self.bank].argmax()
        self.fisher_history += contestants[best_pos]
        return self.bank.pop(best_pos)

# class Fisher3(CAT):
#     def next_item(self):
#         contestants = (self.pdf * self.pdf * (1 - self.pdf)).sum(axis=1) * self.item_features ** 2
#         best_pos = contestants[self.bank].argmax()
#         return self.bank.pop(best_pos)
