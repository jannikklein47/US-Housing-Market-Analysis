from pathlib import Path
import kagglehub

def load_data() -> str:
    """Loads the data into the /data directory and returns the path to the data file."""
    initial_dir = Path('./data/initial')
    initial_dir.mkdir(parents=True, exist_ok=True)

    if not any(initial_dir.iterdir()):
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