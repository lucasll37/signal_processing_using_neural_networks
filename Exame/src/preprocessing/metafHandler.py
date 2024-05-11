from PythonMETAR import Metar
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np
import re


class MetafHandler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f'Started MetafHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _X = X.copy()
        _X["metaf_CAVOK"] = False
        _X["metaf_wind_direction"] = None
        _X["metaf_wind_speed"] = None
        _X["metaf_wind_gust"] = None
        _X["metaf_temperature"] = None
        _X["metaf_dewpoint"] = None
        _X["metaf_qnh"] = None
        _X["metaf_visibility"] = None
        _X["metaf_FEW_alt"] = 0
        _X["metaf_FEW_CB"] = False
        _X["metaf_FEW_TCU"] = False
        _X["metaf_SCT_alt"] = 0
        _X["metaf_SCT_CB"] = False
        _X["metaf_SCT_TCU"] = False
        _X["metaf_BKN_alt"] = 0
        _X["metaf_BKN_CB"] = False
        _X["metaf_BKN_TCU"] = False
        _X["metaf_OVC_alt"] = 0
        _X["metaf_OVC_CB"] = False
        _X["metaf_OVC_TCU"] = False

        for index, row in _X.iterrows():
            if row["metar"] is np.nan:
                continue

            translationMetaf = Metar(row["destino"], row["metaf"])

            _X.loc[index, "metaf_CAVOK"] = bool(re.search("CAVOK", row["metaf"]))

            try:
                wind_direction = translationMetaf.wind["direction"]
                if wind_direction != "VRB":
                    _X.loc[index, "metaf_wind_direction"] = wind_direction
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metaf_wind_speed"] = translationMetaf.wind["speed"]
            except:
                pass  # imputar média

            try:
                gust = translationMetaf.wind["gust"]
                _X.loc[index, "metaf_wind_gust"] = False if gust is None else True
            except:
                _X.loc[index, "metaf_wind_gust"] = False

            try:
                _X.loc[index, "metaf_temperature"] = translationMetaf.temperatures[
                    "temperature"
                ]
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metaf_dewpoint"] = translationMetaf.temperatures[
                    "dewpoint"
                ]
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metaf_qnh"] = translationMetaf.qnh
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metaf_visibility"] = translationMetaf.visibility
            except:
                pass  # imputar média

            if translationMetaf.cloud is not None:
                for cloud in translationMetaf.cloud:

                    try:
                        _X.loc[index, f'metaf_{cloud["code"]}_alt'] = cloud["altitude"]
                    except:
                        pass

                    try:
                        _X.loc[index, f'metaf_{cloud["code"]}_CB'] = cloud["presenceCB"]
                    except:
                        pass

                    try:
                        _X.loc[index, f'metaf_{cloud["code"]}_TCU'] = cloud[
                            "presenceTCU"
                        ]
                    except:
                        pass

        cols = [
            "metaf_wind_direction",
            "metaf_wind_speed",
            "metaf_temperature",
            "metaf_dewpoint",
            "metaf_qnh",
            "metaf_visibility",
        ]

        _X[cols] = _X[cols].fillna(_X[cols].mean())
        print(f'Finished MetafHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return _X
