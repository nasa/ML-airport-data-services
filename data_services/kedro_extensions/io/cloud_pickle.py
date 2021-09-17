from kedro.extras.datasets.pickle import PickleDataSet
import pandas as pd
import cloudpickle
import pickle

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

class PickleDataSetwCloud(PickleDataSet):
    BACKENDS = {"pickle": pickle, "joblib": joblib, "cloudpickle": cloudpickle}
    
