import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from datetime import datetime


class BalancerHandler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        categorical_features,
        method="under",
        ratio=1.0,
        target="target",
        seed=None,
    ):
        
        self.target = target
        self.sampler = None
        self.categorical_features = categorical_features
        self.seed = seed


        if method == "under":
            self.sampler = RandomUnderSampler(
                sampling_strategy=ratio, random_state=self.seed
            )

        elif method == "smote":
            self.sampler = SMOTENC(
                categorical_features=self.categorical_features,
                sampling_strategy=ratio,
                random_state=self.seed,
            )

        else:
            raise ValueError(f"Método de balanceamento '{self.method}' não suportado.")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f'Started BalancerHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _y = X[self.target].copy()
        # _X = X.drop(columns=[self.target])

        _X = X.drop([self.target], axis=1).copy()

        X_resampled, y_resampled = self.sampler.fit_resample(_X, _y)

        df_resampled = pd.DataFrame(X_resampled, columns=_X.columns)
        df_resampled[self.target] = y_resampled
        print(f'Finished BalancerHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return df_resampled







