from kedro.extras.datasets.pandas import CSVDataSet
import pandas as pd

class CSVDataSetEmpty(CSVDataSet) :

    def _load(self) -> pd.DataFrame :
        if self._exists() :
            return super()._load()
        else :
            return pd.DataFrame()
