import pandas as pd
import numpy as np

def dropOutlier(X, lower_percentile=0.025, upper_percentile=0.975):
    keep_rows = pd.Series([True] * X.shape[0], index=X.index)
    name_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    
    for col in name_cols:
        lower_bound = X[col].quantile(lower_percentile)
        upper_bound = X[col].quantile(upper_percentile)
        keep_rows &= (X[col] >= lower_bound) & (X[col] <= upper_bound)
        
    _X = X.loc[keep_rows].copy()

    return _X