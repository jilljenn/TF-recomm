from config import NB_CLASSES
import numpy as np
import random


class CAT:
    def __init__(self, items):
        self.bank = list(items)

    def update_probas(self, cdf, pdf, item_bias):
        self.cdf = cdf
        self.pdf = pdf
        self.item_bias = item_bias


class Random(CAT):
    def next_item(self):
        return self.bank.pop(random.randint(0, len(self.bank) - 1))


class Fisher(CAT):
    def next_item(self):
        arith = np.arange(NB_CLASSES)
        left = (arith ** 2 * self.pdf).sum(axis=1)
        right = (arith * self.pdf).sum(axis=1) ** 2
        contestants = (left - right) * self.item_bias ** 2
        best_pos = contestants[self.bank].argmax()
        return self.bank.pop(best_pos)


class Fisher2(CAT):
    def next_item(self):
        contestants = (self.cdf * (1 - self.cdf)).sum() * self.item_bias ** 2
        best_pos = contestants[self.bank].argmax()
        return self.bank.pop(best_pos)
