import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

def data_split(param_yaml_path):
    with open(param_yaml_path) as params_file:
        params_yaml = yaml.safe_load(params_file)
    git_data = params_yaml['data_source']['git_path']

    df = pd.read_csv(git_data)
    random_state = params_yaml['base']['random_state']
    split_ratio = params_yaml['split']['split_ratio']

    train, test = train_test_split(df, test_size = split_ratio, random_state = random_state)
    
    data_dir = Path(params_yaml['split']['dir'])
    data_dir.mkdir(parents = True, exist_ok = True)

    train_file_path = Path(params_yaml['split']['train_file'])
    train_data_path = data_dir / train_file_path
    train.to_csv(train_data_path, index = False)

    test_data_path = data_dir / Path(params_yaml['split']['test_file'])
    test.to_csv(test_data_path, index = False)



# params = 'D:/Projects/DVC_pipeline/params.yaml'
# with open (params) as p:
#     params_yaml = yaml.safe_load(p)
#     data_dir = Path(params_yaml['split']['dir'])
#     train_file =  data_dir / Path(params_yaml['split']['train_file'])
#     print(train_file)


if __name__ == '__main__':
    data_split(param_yaml_path = 'D:/Projects/DVC_pipeline/params.yaml')