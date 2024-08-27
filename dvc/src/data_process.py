import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def label_encode(data_path, target_col):
    df = pd.read_csv(data_path)
    le = LabelEncoder()

    df[target_col] = le.fit_transform(df[target_col])
    return df

# d = pd.DataFrame({'a' : ['p', 'b', 'o'], 'col' : ['to', 'from', 'to']})
# le = LabelEncoder()
# d['col'] = le.fit_transform(d['col'])
# print(d)

if __name__ == '__main__':
    param_yaml_path = 'D:/Projects/DVC_pipeline/params.yaml'
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    data_dir = Path(params_yaml['split']['dir'])
    train_file_path = Path(params_yaml['split']['train_file'])
    train_data_path = data_dir / train_file_path
    processed_train_data = label_encode(data_path = train_data_path, target_col = params_yaml['base']['target_col'])
    
    processed_data_dir = Path(params_yaml['process']['dir'])
    processed_data_dir.mkdir(parents = True, exist_ok = True)
    processed_train_data_path = processed_data_dir / train_file_path
    processed_train_data.to_csv(processed_train_data_path, index = False)
    
    test_file_path = Path(params_yaml['split']['test_file'])
    test_data_path = data_dir / test_file_path
    processed_test_data = label_encode(data_path = test_data_path, target_col = params_yaml['base']['target_col'])
    
    processed_test_data_path = processed_data_dir / test_file_path
    processed_test_data.to_csv(processed_test_data_path, index = False)