
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class TransformedTargetClassifier(TransformedTargetRegressor):
    def __init__(self, regressor=None, *, transformer=None,
                 func=None, inverse_func=None, check_inverse=True):
        super().__init__(
            regressor=regressor,
            transformer=transformer,
            func=func,
            inverse_func=inverse_func,
            check_inverse=check_inverse,
            )
        self.classes_ = None

    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters passed to the ``fit`` method of the underlying
            regressor.


        Returns
        -------
        self : object
        """
        # y = check_array(y, accept_sparse=False, force_all_finite=True,
        #                 ensure_2d=False, dtype='numeric')
        if isinstance(y, pd.Series):
            y = y.values

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from ..linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        self.classes_ = self.transformer_.classes_

        return self

    def predict_proba(
            self,
            X,
            ):
        return self.regressor_.predict_proba(X)

    def score(self,
              X,
              y,
              sample_weight=None,
              ):
        return accuracy_score(
            y,
            self.regressor_.predict(X),
            sample_weight=sample_weight,
            )
