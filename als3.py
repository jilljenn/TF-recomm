from sklearn.metrics import mean_squared_error
from scipy.sparse import coo_matrix
from collections import defaultdict
import pickle
import numpy as np
import os.path


class MangakiALS3:
    def __init__(self, nb_components=20, nb_iterations=20, lambda_=0.1):
        super().__init__()
        self.nb_components = nb_components
        self.nb_iterations = nb_iterations
        self.lambda_ = lambda_

    @property
    def is_serializable(self):
        return True

    def fit(self, X, y, y_test, X_test):
        self.X_test = X_test
        self.y_test = y_test
        self.init_vars()
        self.bias = y.mean()
        #y -= self.bias
        self.matrix, self.matrixT = self.to_dict(X, y)
        self.ratings_of_user, self.ratings_of_work = self.to_sparse(X, y)
        users, works = map(np.unique, self.ratings_of_user.nonzero())
        for nb_iter in range(self.nb_iterations):
            print('Step', nb_iter, self.compute_rmse(self.y_test, self.predict(self.X_test)))
            for user_id in users:
                self.fit_user(user_id)
            for work_id in works:
                self.fit_work(work_id)

    def to_dict(self, X, y):
        matrix = defaultdict(dict)
        matrixT = defaultdict(dict)
        wou = defaultdict(list)
        rou = defaultdict(list)
        uow = defaultdict(list)
        row = defaultdict(list)
        for (user_id, work_id), rating in zip(X, y):
            # matrix[user_id][work_id] = rating
            # matrixT[work_id][user_id] = rating
            wou[user_id].append(work_id)
            rou[user_id].append(rating)
            uow[work_id].append(user_id)
            row[work_id].append(rating)
        self.wou = wou
        self.rou = rou
        self.uow = uow
        self.row = row
        return matrix, matrixT

    def to_sparse(self, X, y):
        user_ids, work_ids = zip(*X)  # Columns of X
        ratings = coo_matrix((y, (user_ids, work_ids)), shape=(self.nb_users, self.nb_works))
        return ratings.tocsr(), ratings.tocsc()

    def init_vars(self):
        self.U = np.random.rand(self.nb_users, self.nb_components)
        self.V = np.random.rand(self.nb_works, self.nb_components)
        # self.U = np.random.multivariate_normal(mean=np.zeros(self.nb_components), cov=np.eye(self.nb_components), size=self.nb_users)
        # self.V = np.random.multivariate_normal(mean=np.zeros(self.nb_components), cov=np.eye(self.nb_components), size=self.nb_works)
        self.W_user = np.random.rand(self.nb_users)#np.random.normal(size=self.nb_users)
        self.W_work = np.random.rand(self.nb_works)#np.random.normal(size=self.nb_works)
        self.bias = 0

    def fit_user(self, user_id):
        #Ji = self.ratings_of_user[user_id].indices
        #Ri = self.ratings_of_user[user_id].data
        #Ji = list(self.matrix[user_id].keys())
        #Ri = list(self.matrix[user_id].values())
        Ji = self.wou[user_id]
        Ri = self.rou[user_id]
        Ni = len(Ji)
        Vi = self.V[Ji]
        Wi = self.W_work[Ji]
        bi = self.W_user[user_id] + self.bias
        Li = self.lambda_ * Ni * np.eye(self.nb_components)
        # print(Ri.shape)
        # print(Wi.shape)
        # print((Ri - Wi - bi).shape)
        # print(Vi.shape)
        # print(Vi.dot(Ri - Wi - bi).shape)
        # print(Vi)
        # print(np.linalg.det(Vi.T.dot(Vi) + Li))
        self.U[user_id] = np.linalg.solve(Vi.T.dot(Vi) + Li, (Ri - Wi - bi).dot(Vi))
        #print(Vi.T.dot(Vi) + Li)
        #self.W_user[user_id] = (Ri - self.M[user_id, Ji]).mean() / (1 + self.lambda_)
        self.W_user[user_id] = (Ri - self.U[user_id].dot(Vi.T) - Wi).mean() / (1 + self.lambda_) - self.bias

    def fit_work(self, work_id):
        # Ij = self.ratings_of_work[:, work_id].indices
        # Rj = self.ratings_of_work[:, work_id].data
        #Ij = list(self.matrixT[work_id].keys())
        #Rj = list(self.matrixT[work_id].values())
        Ij = self.uow[work_id]
        Rj = self.row[work_id]
        Nj = len(Ij)
        Uj = self.U[Ij]
        Wj = self.W_user[Ij]
        bj = self.W_work[work_id] + self.bias
        Lj = self.lambda_ * Nj * np.eye(self.nb_components)
        self.V[work_id] = np.linalg.solve(Uj.T.dot(Uj) + Lj, (Rj - Wj - bj).dot(Uj))
        #self.W_work[work_id] = (Rj - self.M[Ij, work_id]).mean() / (1 + self.lambda_)
        self.W_work[work_id] = (Rj - self.V[work_id].dot(Uj.T) - Wj).mean() / (1 + self.lambda_) - self.bias

    def predict(self, X):
        user_ids, work_ids = zip(*X)
        self.M = self.U.dot(self.V.T) + self.W_user.reshape(-1, 1) + self.W_work.reshape(1, -1) + self.bias
        return self.M[user_ids, work_ids]

    def get_shortname(self):
        return 'als3-%d' % self.nb_components

    @staticmethod
    def compute_rmse(y_pred, y_true):
        return mean_squared_error(y_true, y_pred) ** 0.5

    def save(self, filename):
        if not self.is_serializable:
            raise NotImplementedError
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.size = os.path.getsize(filename)  # In bytes

    def load(self, filename):
        """
        This function raises FileNotFoundException if no backup exists.
        """
        if not self.is_serializable:
            raise NotImplementedError
        with open(filename, 'rb') as f:
            backup = pickle.load(f)
        return backup

    def compute_all_errors(self, X_train, y_train, X_test, y_test):
        y_train_pred = self.predict(X_train)
        print('Train RMSE=%f' % self.compute_rmse(y_train, y_train_pred))
        y_test_pred = self.predict(X_test)
        print('Test RMSE=%f' % self.compute_rmse(y_test, y_test_pred))
