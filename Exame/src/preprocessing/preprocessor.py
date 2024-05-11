import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from .inputerHandler import ImputeHandler
from .metarHandler import MetarHandler
from .metafHandler import MetafHandler
from .satelliteImageHandler import SatelliteImageHandler
from .dataHandler import DateHandler
from .balancerHandler import BalancerHandler
from .outlierHandler import dropOutlier


def preprocessor(
    df,
    satImage_width=20,
    satImage_outputWidth=32,
    satImage_outputHeight=32,
    satImage_printRoutes=False,
    satImage_printEachImage=False,
    date_holiday_near=2,
    balancer_method="under",
    balancer_ratio=0.9,
    balancer_target="espera",
    balancer_categorical_features=[0, 1],
    apply_balancer=True,
    drop_outlier=True
):

    steps = list()

    if apply_balancer:
        steps.append(
            (
                "balancer",
                BalancerHandler(
                    method=balancer_method,
                    ratio=balancer_ratio,
                    target=balancer_target,
                    categorical_features=balancer_categorical_features,
                ),
            )
        )

    steps.extend(
        [
            ("imputer", ImputeHandler()),
            ("metar", MetarHandler()),
            ("metaf", MetafHandler()),
            (
                "satImage",
                SatelliteImageHandler(
                    width=satImage_width,
                    outputWidth=satImage_outputWidth,
                    outputHeight=satImage_outputHeight,
                    printRoutes=satImage_printRoutes,
                    printEachImage=satImage_printEachImage,
                ),
            ),
            ("date", DateHandler(holiday_near=date_holiday_near)),
        ]
    )

    pipeline_preprocessor = Pipeline(steps)

    df = pipeline_preprocessor.fit_transform(df)

    df.drop(
        [
            "metar",
            "metaf",
            "hora_ref",
            "origem",
            "url_img_satelite",
            "metar_FEW_TCU",
            "metar_SCT_TCU",
            "metar_BKN_CB",
            "metar_BKN_TCU",
            "metar_OVC_CB",
            "metar_OVC_TCU",
            "metaf_FEW_TCU",
            "metaf_SCT_TCU",
            "metaf_BKN_CB",
            "metaf_BKN_TCU",
            "metaf_OVC_CB",
            "metaf_OVC_TCU",
        ],
        axis=1,
        inplace=True,
    )

    if drop_outlier:
        df = dropOutlier(df)

    df.reindex(
        columns=[c for c in df.columns if c != balancer_target] + [balancer_target]
    )

    return df
