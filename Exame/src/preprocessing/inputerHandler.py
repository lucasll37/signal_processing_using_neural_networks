from datetime import datetime, timedelta
import requests
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class ImputeHandler(BaseEstimator, TransformerMixin):

    def __init__(self, key="Q6USHGkfCPPyILmtVa49gIJTeK7Gxi5uYVRMFQMo"):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f'Started ImputeHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _X = X.copy()

        ########### METAR ###############
        for index, row in _X[_X["metar"].isna()].iterrows():
            loc = row["destino"]
            date = datetime.strptime(row["hora_ref"], "%Y-%m-%dT%H:%M:%SZ")
            str_date = date.strftime("%Y%m%d%H")

            url = f"https://api-redemet.decea.mil.br/mensagens/metar/{loc}?api_key={self.key}&data_ini={date}&data_fim={date}"

            try:
                raise Exception(f"Erro na requisição")

                # response = requests.request("GET", url)

                # if response.status_code == 200:
                #     metar_data = response.json()
                #     _X.loc[index, 'metar'] = metar_data['data']['data'][0]['mens']

                # else:
                #     print(f"Erro: {response.status_code}")

            except:
                pass

        _X["metar"].fillna(_X["metar"].mode(), inplace=True)

        ########### METAF ###############
        for index, row in _X[_X["metaf"].isna()].iterrows():
            loc = row["destino"]
            date = datetime.strptime(row["hora_ref"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(
                hours=1
            )
            str_date = date.strftime("%Y%m%d%H")

            url = f"https://api-redemet.decea.mil.br/mensagens/metar/{loc}?api_key={self.key}&data_ini={str_date}&data_fim={str_date}"

            try:
                raise Exception(f"Erro na requisição")

                # response = requests.request("GET", url)

                # if response.status_code == 200:
                #     metar_data = response.json()
                #     _X.loc[index, 'metaf'] = metar_data['data']['data'][0]['mens']

                # else:
                #     print(f"Erro: {response.status_code}")

            except:
                _X.loc[index, "metaf"] = row["metar"]

        _X["metaf"].fillna(_X["metar"].mode(), inplace=True)

        print(f'Finished ImputeHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return _X
