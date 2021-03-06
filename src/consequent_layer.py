from abc import ABC, abstractmethod

import numpy as np
import torch
#from torch.linalg import lstsq

dtype = torch.float


class ConsequentLayerType:
    HYBRID = 0
    PLAIN = 1
    SYMMETRIC = 2
    MAMDANI = 4


class AbstractConsequentLayer(torch.nn.Module, ABC):
    def __init__(self, coeff):
        super().__init__()
        self._coeff = coeff

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
        """
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        """
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        """
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    @abstractmethod
    def fit_coeff(self, *params):
        pass


class ConsequentLayer(AbstractConsequentLayer):
    """
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    """

    def __init__(self, d_in, d_rule, d_out, dtype):
        super(ConsequentLayer, self).__init__(
            torch.zeros(torch.Size([d_rule, d_out, d_in + 1]), dtype=dtype, requires_grad=True))

        self.ones_cache = dict()

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        """
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        """
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        """
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        """
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        """
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            # coeff_2d = lstsq(weighted_x_2d, y_actual_2d).solution
            coeff_2d = lstsq(weighted_x_2d, y_actual_2d).solution
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1] + 1, -1) \
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        """
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        """
        x_shape = x.shape[0]
        if x_shape not in self.ones_cache:
            ones = torch.ones(x_shape, 1)
            self.ones_cache[x_shape] = ones
        else:
            ones = self.ones_cache[x_shape]

        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, ones], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    """
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    """

    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        """
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        """
        """
        assert False, \
            'Not hybrid learning: I\'m using BP to learn coefficients'


class SymmetricWeightsConsequentLayer(AbstractConsequentLayer):
    """
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    """

    def __init__(self, d_in, d_rule, d_out, dtype):
        super().__init__(torch.zeros(torch.Size([int(np.ceil(d_rule / 2)), d_out, d_in + 1]), dtype=dtype))
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff, requires_grad=True))

        self.permutation_cache = dict()
        self.ones_cache = dict()

    def get_permutation(self):
        n = self.coeff.shape[0]

        if n in self.permutation_cache:
            return self.permutation_cache[n]
        else:
            p = self.calculate_permutation_matrix(n)
            self.permutation_cache[n] = p
            return p

    def calculate_permutation_matrix(self, n):
        with torch.no_grad():
            xs = []
            ys = []
            values = []

            new_n = 2 * n - 1

            for i in range(n - 1):
                xs.append(i)
                ys.append(i)
                values.append(-1)

            for i in range(n):
                xs.append(new_n - i - 1)
                ys.append(i)
                values.append(1)

            return (torch.sparse_coo_tensor(indices=torch.tensor([xs, ys]),
                                            values=torch.tensor(values),
                                            size=[new_n, n], dtype=self.coeff.dtype), (new_n, n))

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        """
        return self.coefficients

    @coeff.setter
    def coeff(self, new_coeff):
        """
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        """
        self.coefficients = new_coeff

    def fit_coeff(self):
        """
        """
        # summ = self.coeff.abs().sum(dim=2).view(-1)
        #
        # c = torch.masked_select(summ,summ >= 1e-9)
        #
        # print(c.mean(), c.median(), c.std(), c.mean() - 3 * c.std())
        #
        # mask = torch.greater_equal(summ, c.mean() - 3 * c.std())

        # summ = self.coeff.abs().sum(dim=2).view(-1)
        # summ = torch.masked_select(summ, summ >= 1e-9)
        #
        # q = torch.tensor([.25, 0.75], dtype=summ.dtype)
        # q1, q2 = torch.quantile(summ, q)
        #
        # IQR = q2 - q1
        # print(q1, q2, q1 - 1.5 * IQR)
        # mask = torch.greater_equal(summ, q1 - 1.5 * IQR)

        # mask = torch.greater_equal(summ, c.mean() - 3 * c.std())

        # TODO maybe make it so that instead of removing everything at oncee, remove the elements one after the after after randomly
        mask = torch.greater_equal(self.coeff.abs().sum(dim=2), 1e-9).view(-1)

        assert not torch.all(~mask), "Error, all the coefficients have been removed, nothing has trained"

        update = torch.any(~mask)

        if update:
            # FIXME find a better way to
            # self.coeff.data = self.coeff[mask].data
            # print(list(self.named_parameters()))
            self.register_parameter('coefficients', torch.nn.Parameter(self.coeff[mask], requires_grad=True))

        return mask, update

    def batch_bmm(self, matrix, matrix_batch):
        """
        From: https://github.com/pytorch/pytorch/issues/14489
        :param matrix: Sparse or dense matrix, size (m, n).
        :param matrix_batch: Batched dense matrices, size (b, n, k).
        :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
        """

        matrix, (rows, cols) = matrix

        batch_size = matrix_batch.shape[1]
        # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
        dimensions = cols
        vectors = matrix_batch.reshape(dimensions, -1)

        # A matrix-matrix product is a batched matrix-vector product of the columns.
        # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
        return matrix.mm(vectors).reshape(rows, batch_size, -1)

    def forward(self, x):
        """
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        """
        x_shape = x.shape[0]
        if x_shape not in self.ones_cache:
            ones = torch.ones(x_shape, 1)
            self.ones_cache[x_shape] = ones
        else:
            ones = self.ones_cache[x_shape]

        P = self.get_permutation()

        symetric_coeff = self.batch_bmm(P, self.coeff)

        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, ones], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(symetric_coeff, x_plus.t())

        return y_pred.transpose(0, 2)  # swaps cases and rules


class MamdaniConsequentLayer(torch.nn.Module):
    def __init__(self, mamdani_defs, output_membership_mapping):
        super().__init__()
        self.mamdani_defs = mamdani_defs
        self.output_membership_mapping = output_membership_mapping

    def forward(self, x):
        self.mamdani_defs.cache()

        # FIXME make it work for multiple outputs
        # output = list(self.mamdani_defs[membership_id[0]] for membership_id in self.output_membership_mapping)

        # data = torch.stack([torch.stack([self.mamdani_defs[var] for var in membership_id]) for membership_id in self.output_membership_mapping]).transpose(0,1)

        # ordered_dict = OrderedDict()

        # for membership_id in self.output_membership_mapping:
        #     for i, var in enumerate(membership_id):
        #         ordered_dict.setdefault(i, [])
        #
        #         ordered_dict[i].append(self.mamdani_defs[var])
        #
        # torch.stack(ordered_dict.values())

        # data = torch.stack(
        #     tuple(
        #         torch.stack(tuple(self.mamdani_defs[var] for var in membership_id))
        #         for membership_id in self.output_membership_mapping))
        # print(self.output_membership_mapping)
        # print(self.mamdani_defs[6])
        # print(self.mamdani_defs[5])
        # print(self.mamdani_defs[4])
        # print(self.mamdani_defs[3])
        # print(self.mamdani_defs[2])
        # print(self.mamdani_defs[1])
        # print(self.mamdani_defs[0])

        data = torch.stack(
            [self.mamdani_defs[membership_id[0]] for membership_id in self.output_membership_mapping]).unsqueeze(1)
        # print(data)
        # dsad

        return data
