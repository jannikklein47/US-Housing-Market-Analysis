import os
import pandas as pd
from load_data import get_initial_data_path

df = pd.read_csv(get_initial_data_path()[0])

os.makedirs("data/final", exist_ok=True)
df.to_csv("data/final/data.csv", index=False)
