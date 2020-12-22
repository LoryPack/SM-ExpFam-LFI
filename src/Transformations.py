import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class LowerBoundedVarScaler(MinMaxScaler):
    """
        We implement first the lower bound transformation with the log. Afterwards, we still optionally rescale
        the transformed variables to the (0,1) range (default for this is True)
        """

    def __init__(self, lower_bound=0, feature_range=(0, 1), copy=True, rescale_transformed_vars=True):
        # lower_bound can be both scalar of array-like with size the size of the variable; if not provided, it will be assumed to be 0
        self.lower_bound = lower_bound
        self.feature_range = feature_range
        self.copy = copy
        self.rescale_transformed_vars = rescale_transformed_vars

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        # assert (X > self.lower_bound).all()
        if np.any(X <= self.lower_bound):
            raise RuntimeError("The provided data are out of the bounds.")

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = np.log(X - self.lower_bound)

        if self.rescale_transformed_vars:
            return super().fit(X)
        else:
            return self

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        # assert (X > self.lower_bound).all()
        if np.any(X <= self.lower_bound):
            raise RuntimeError("The provided data is out of the bounds.")

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = np.log(X - self.lower_bound)
        if self.rescale_transformed_vars:
            return super().transform(X)
        else:
            return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.rescale_transformed_vars:
            X = super().inverse_transform(X)
        else:
            X = X

        # now apply the inverse transform
        return np.exp(X) + self.lower_bound

    def jac_log_det(self, x):
        """Returns the log determinant of the Jacobian: log |J_t(x)|.

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the original space (with lower bound constraints).
        Returns
        -------
        res : float
            log determinant of the jacobian
        """
        if np.any(x <= self.lower_bound):
            raise RuntimeError("The provided data is out of the bounds.")

        return - np.sum(np.log(x - self.lower_bound))

    def jac_log_det_inverse_transform(self, x):
        """Returns the log determinant of the Jacobian evaluated in the inverse transform:
        log |J_t(t^{-1}(x))| = - log |J_{t^{-1}}(x)|

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the transformed space (spanning the whole R^d).
        Returns
        -------
        res : float
            log determinant of the jacobian evaluated in t^{-1}(x)
        """
        return - x.sum()


class TwoSidedBoundedVarScaler(MinMaxScaler):
    """
    We implement first the lower bound transformation with the log. Afterwards, we still optionally rescale
    the transformed variables to the (0,1) range (default for this is True)
    """

    def __init__(self, lower_bound=0, upper_bound=1, feature_range=(0, 1), copy=True,
                 rescale_transformed_vars=True):

        # upper and lower bounds can be both scalar of array-like with size the size of the variable; if not provided, they will be assumed to be 0 and 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.feature_range = feature_range
        self.copy = copy
        self.rescale_transformed_vars = rescale_transformed_vars

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1 - x)

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        # assert (X > self.lower_bound).all() and (X < self.upper_bound).all()
        if (X <= self.lower_bound).any() or (X >= self.upper_bound).any():
            raise RuntimeError("The provided data is out of the bounds.")

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = self.logit((X - self.lower_bound) / (self.upper_bound - self.lower_bound))

        if self.rescale_transformed_vars:
            return super().fit(X)
        else:
            return self

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        # assert (X > self.lower_bound).all() and (X < self.upper_bound).all()
        if (X <= self.lower_bound).any() or (X >= self.upper_bound).any():
            # print(X)
            raise RuntimeError("The provided data are out of the bounds.")

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = self.logit((X - self.lower_bound) / (self.upper_bound - self.lower_bound))
        if self.rescale_transformed_vars:
            return super().transform(X)
        else:
            return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.rescale_transformed_vars:
            X = super().inverse_transform(X)
        else:
            X = X

        # now apply the inverse transform
        return (self.upper_bound - self.lower_bound) * np.exp(X) / (1 + np.exp(X)) + self.lower_bound

    def jac_log_det(self, x):
        """Returns the log determinant of the Jacobian: log |J_t(x)|.

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the original space (with lower bound constraints).
        Returns
        -------
        res : float
            log determinant of the jacobian
        """
        if (x <= self.lower_bound).any() or (x >= self.upper_bound).any():
            raise RuntimeError("The provided data are out of the bounds.")

        return np.sum(np.log((self.upper_bound - self.lower_bound) / ((x - self.lower_bound) * (self.upper_bound - x))))

    def jac_log_det_inverse_transform(self, x):
        """Returns the log determinant of the Jacobian evaluated in the inverse transform:
        log |J_t(t^{-1}(x))| = - log |J_{t^{-1}}(x)|

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the transformed space (spanning the whole R^d).
        Returns
        -------
        res : float
            log determinant of the jacobian evaluated in t^{-1}(x)
        """

        res_a = np.log(self.upper_bound - self.lower_bound)
        indices = x < 100  # for avoiding numerical overflow
        res_b = np.copy(x)
        res_b[indices] = np.log(1 + np.exp(x[indices]))

        indices = x > - 100  # for avoiding numerical overflow
        res_c = np.copy(- x)
        res_c[indices] = np.log(1 + np.exp(- x[indices]))

        res = np.sum(res_b + res_c - res_a)
        # res = - np.sum(np.log((self.upper_bound - self.lower_bound) / ((1 + np.exp(x)) * (1 + np.exp(-x)))))

        return res


class BoundedVarScaler(MinMaxScaler):
    """
    This scaler implements both lower bounded and two sided bounded transformations according to the provided bounds;
    After the nonlinear transformation is applied, we still optionally rescale the transformed variables to the (0,1)
    range (default for this is True)
    """

    def __init__(self, lower_bound, upper_bound, feature_range=(0, 1), copy=True, rescale_transformed_vars=True):

        # upper and lower bounds can be both scalar or array-like with size the size of the variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if not hasattr(lower_bound, "shape") or not hasattr(upper_bound, "shape"):
            raise RuntimeError("Provided lower and upper bounds need to be arrays.")
        elif hasattr(lower_bound, "shape") and hasattr(upper_bound, "shape") and lower_bound.shape != upper_bound.shape:
            raise RuntimeError("Provided lower and upper bounds need to have same shape.")

        # note that == None checks if the array is None element wise.
        self.unbounded_vars = np.logical_and(np.equal(lower_bound, None), np.equal(upper_bound, None))
        self.lower_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.equal(upper_bound, None))
        self.upper_bounded_vars = np.logical_and(np.equal(lower_bound, None), np.not_equal(upper_bound, None))
        self.two_sided_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.not_equal(upper_bound, None))
        if self.upper_bounded_vars.any():
            raise NotImplementedError("We do not yet implement the transformation for upper bounded random variables")

        self.lower_bound_lower_bounded = self.lower_bound[self.lower_bounded_vars].astype("float32")
        self.lower_bound_two_sided = self.lower_bound[self.two_sided_bounded_vars].astype("float32")
        self.upper_bound_two_sided = self.upper_bound[self.two_sided_bounded_vars].astype("float32")

        self.feature_range = feature_range
        self.copy = copy
        self.rescale_transformed_vars = rescale_transformed_vars

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1 - x)

    def _check_data_in_bounds(self, X):
        if np.any(X[:, self.lower_bounded_vars] <= self.lower_bound_lower_bounded):
            raise RuntimeError("The provided data are out of the bounds.")
        if (X[:, self.two_sided_bounded_vars] <= self.lower_bound[self.two_sided_bounded_vars]).any() or (
                X[:, self.two_sided_bounded_vars] >= self.upper_bound_two_sided).any():
            raise RuntimeError("The provided data is out of the bounds.")

    def _apply_nonlinear_transf(self, X):
        # apply the different scalers to the different kind of variables:
        X_transf = X.copy()
        X_transf[:, self.lower_bounded_vars] = np.log(X[:, self.lower_bounded_vars] - self.lower_bound_lower_bounded)
        X_transf[:, self.two_sided_bounded_vars] = self.logit(
            (X[:, self.two_sided_bounded_vars] - self.lower_bound_two_sided) / (
                    self.upper_bound_two_sided - self.lower_bound_two_sided))
        return X_transf

    def _check_reshape_single_sample(self, x):
        if len(x.shape) == 1:
            pass
        elif len(x.shape) == 2 and x.shape[0] == 1:
            x = x.reshape(-1)
        else:
            raise RuntimeError("This can be computed for one sample at a time.")
        return x

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        self._check_data_in_bounds(X)

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = self._apply_nonlinear_transf(X)

        if self.rescale_transformed_vars:
            return super().fit(X)
        else:
            return self

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # need to check if we can apply the log first:
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        self._check_data_in_bounds(X)

        # we first transform the data with the log transformation and then apply the scaler (optionally):
        X = self._apply_nonlinear_transf(X)

        if self.rescale_transformed_vars:
            return super().transform(X)
        else:
            return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.rescale_transformed_vars:
            X = super().inverse_transform(X)
        else:
            X = X

        # now apply the inverse transform
        inv_X = X.copy()
        inv_X[:, self.two_sided_bounded_vars] = (self.upper_bound_two_sided - self.lower_bound_two_sided) * np.exp(
            X[:, self.two_sided_bounded_vars]) / (1 + np.exp(
            X[:, self.two_sided_bounded_vars])) + self.lower_bound_two_sided
        inv_X[:, self.lower_bounded_vars] = np.exp(X[:, self.lower_bounded_vars]) + self.lower_bound_lower_bounded

        return inv_X

    def jac_log_det(self, x):
        """Returns the log determinant of the Jacobian: log |J_t(x)|.

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the original space (with lower bound constraints).
        Returns
        -------
        res : float
            log determinant of the jacobian
        """
        x = self._check_reshape_single_sample(x)
        self._check_data_in_bounds(x.reshape(1, -1))

        results = np.zeros_like(x)
        results[self.two_sided_bounded_vars] = np.log(
            (self.upper_bound_two_sided - self.lower_bound_two_sided).astype("float64") / (
                    (x[self.two_sided_bounded_vars] - self.lower_bound_two_sided) * (
                    self.upper_bound_two_sided - x[self.two_sided_bounded_vars])))
        results[self.lower_bounded_vars] = - np.log(x[self.lower_bounded_vars] - self.lower_bound_lower_bounded)

        return np.sum(results)

    def jac_log_det_inverse_transform(self, x):
        """Returns the log determinant of the Jacobian evaluated in the inverse transform:
        log |J_t(t^{-1}(x))| = - log |J_{t^{-1}}(x)|

        Note that this considers only the Jacobian arising from the non-linear transformation, neglecting the scaling
        term arising from the subsequent linear rescaling. In fact, the latter does not play any role in MCMC acceptance
        rate.

        Parameters
        ----------
        x : array-like of shape (n_features)
            Input data, living in the transformed space (spanning the whole R^d).
        Returns
        -------
        res : float
            log determinant of the jacobian evaluated in t^{-1}(x)
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        x = self._check_reshape_single_sample(x)

        results = np.zeros_like(x)
        results[self.lower_bounded_vars] = - x[self.lower_bounded_vars]
        # two sided: need some tricks to avoid numerical issues:
        results[self.two_sided_bounded_vars] = - np.log(
            self.upper_bound_two_sided - self.lower_bound_two_sided)

        indices = x[self.two_sided_bounded_vars] < 100  # for avoiding numerical overflow
        res_b = np.copy(x)[self.two_sided_bounded_vars]
        res_b[indices] = np.log(1 + np.exp(x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_b

        indices = x[self.two_sided_bounded_vars] > - 100  # for avoiding numerical overflow
        res_c = np.copy(- x)[self.two_sided_bounded_vars]
        res_c[indices] = np.log(1 + np.exp(- x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_c

        # res = res_b + res_c - res_a

        return np.sum(results)
