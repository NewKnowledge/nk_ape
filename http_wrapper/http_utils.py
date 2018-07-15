''' Utility functions for json '''
from flask import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class PandasEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return [row.to_dict() for _, row in obj.iterrows()]
        return NumpyEncoder.default(self, obj)
