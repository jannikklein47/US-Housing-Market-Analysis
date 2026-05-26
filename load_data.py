from pathlib import Path
import kagglehub
import os 

def load_data() -> str:
    """Loads the data into the /data directory and returns the path to the data file."""
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    if not os.listdir('./data'):
        kagglehub.dataset_download('ahmedshahriarsakib/usa-real-estate-dataset', output_dir='./data/initial/data.csv')

    path_obj = Path('./data/initial')

    return [file.as_posix() for file in path_obj.rglob('*.csv') if file.is_file()]

def get_initial_data_path() -> str:
    path_obj = Path('./data/initial')

    return [file.as_posix() for file in path_obj.rglob('*.csv') if file.is_file()]

def get_final_data_path() -> str:
    path_obj = Path('./data/final')

    return [file.as_posix() for file in path_obj.rglob('*.csv') if file.is_file()]
load_data()