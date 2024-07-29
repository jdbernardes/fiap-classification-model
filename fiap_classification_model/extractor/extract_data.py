import pandas as pd
from ucimlrepo import fetch_ucirepo


class ExtractUCIData:
    def __init__(self, source_data_id: int) -> None:
        """
        Param _source_data_id: this represents the UCI repo ID
        Ex: 545 - Rice (Cammeo and Osmancik), 186 - Wine Quality
        """
        self._uci_repo = fetch_ucirepo(id=source_data_id)

    def retrieve_x(self) -> pd.DataFrame:
        return self._uci_repo.data.features

    def retrieve_y(self) -> pd.DataFrame:
        return self._uci_repo.data.targets

    def read_data(self) -> pd.DataFrame:
        df_x = self.retrieve_x()
        df_y = self.retrieve_y()
        df = df_x.copy()
        df["Class"] = df_y
        return df

    def save_csv(self) -> None:
        df = self.read_data()
        df.to_csv("./Data/souce.csv")


if __name__ == "__main__":
    data = ExtractUCIData(545)
    data.save_csv()
