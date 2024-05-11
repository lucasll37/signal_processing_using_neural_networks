import pandas as pd

from datetime import datetime, timedelta
import holidays

br_holidays = holidays.Brazil()

from sklearn.base import BaseEstimator, TransformerMixin


class DateHandler(BaseEstimator, TransformerMixin):

    def __init__(self, holiday_near=2):
        self.holiday_near = holiday_near

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f'Started DateHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _X = X.copy()
        _X["hora_ref"] = pd.to_datetime(_X["hora_ref"])
        _X["hour"] = _X["hora_ref"].dt.hour
        _X["weekday"] = _X["hora_ref"].dt.dayofweek
        _X["mounth"] = _X["hora_ref"].dt.month
        _X["holiday_near"] = _X["hora_ref"].apply(self._is_holiday_near)

        print(f'Finished DaterHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return _X

    def _is_holiday_near(self, date):
        for delta in range(-self.holiday_near, self.holiday_near + 1):
            if date + timedelta(days=delta) in br_holidays:
                return True
        return False

    # def get_params(self, deep=True):

    #     return {
    #         "holiday_near": self.holiday_near
    #     }

    # def set_params(self, **params):

    #     valid_params = self.get_params(deep=True)

    #     for key, value in params.items():
    #         if key not in valid_params:
    #             raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}. Check the list of available parameters with `estimator.get_params().keys()`.")

    #         setattr(self, key, value)

    #     return self
