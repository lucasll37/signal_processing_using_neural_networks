from PythonMETAR import Metar
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np
import re


class MetarHandler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f'Started MetarHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _X = X.copy()
        _X["metar_CAVOK"] = False
        _X["metar_wind_direction"] = None
        _X["metar_wind_speed"] = None
        _X["metar_wind_gust"] = None
        _X["metar_temperature"] = None
        _X["metar_dewpoint"] = None
        _X["metar_qnh"] = None
        _X["metar_visibility"] = None
        _X["metar_FEW_alt"] = 0
        _X["metar_FEW_CB"] = False
        _X["metar_FEW_TCU"] = False
        _X["metar_SCT_alt"] = 0
        _X["metar_SCT_CB"] = False
        _X["metar_SCT_TCU"] = False
        _X["metar_BKN_alt"] = 0
        _X["metar_BKN_CB"] = False
        _X["metar_BKN_TCU"] = False
        _X["metar_OVC_alt"] = 0
        _X["metar_OVC_CB"] = False
        _X["metar_OVC_TCU"] = False

        for index, row in _X.iterrows():
            if row["metar"] is np.nan:
                continue

            translationMetar = Metar(row["destino"], row["metar"])

            _X.loc[index, "metar_CAVOK"] = bool(re.search("CAVOK", row["metar"]))

            try:
                wind_direction = translationMetar.wind["direction"]
                if wind_direction != "VRB":
                    _X.loc[index, "metar_wind_direction"] = wind_direction
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metar_wind_speed"] = translationMetar.wind["speed"]
            except:
                pass  # imputar média

            try:
                gust = translationMetar.wind["gust"]
                _X.loc[index, "metar_wind_gust"] = False if gust is None else True
            except:
                _X.loc[index, "metar_wind_gust"] = False

            try:
                _X.loc[index, "metar_temperature"] = translationMetar.temperatures[
                    "temperature"
                ]
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metar_dewpoint"] = translationMetar.temperatures[
                    "dewpoint"
                ]
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metar_qnh"] = translationMetar.qnh
            except:
                pass  # imputar média

            try:
                _X.loc[index, "metar_visibility"] = translationMetar.visibility
            except:
                pass  # imputar média

            if translationMetar.cloud is not None:
                for cloud in translationMetar.cloud:

                    try:
                        _X.loc[index, f'metar_{cloud["code"]}_alt'] = cloud["altitude"]
                    except:
                        pass

                    try:
                        _X.loc[index, f'metar_{cloud["code"]}_CB'] = cloud["presenceCB"]
                    except:
                        pass

                    try:
                        _X.loc[index, f'metar_{cloud["code"]}_TCU'] = cloud[
                            "presenceTCU"
                        ]
                    except:
                        pass

        cols = [
            "metar_wind_direction",
            "metar_wind_speed",
            "metar_temperature",
            "metar_dewpoint",
            "metar_qnh",
            "metar_visibility",
        ]

        _X[cols] = _X[cols].fillna(_X[cols].mean())
        print(f'Finished MetarHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return _X
