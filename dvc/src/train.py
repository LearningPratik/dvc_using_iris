import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle

def training(param_yaml_path):
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    
    processed_data_dir = Path(params_yaml['process']['dir'])
    train_file_path = Path(params_yaml['process']['train_file'])
    train_data_path = processed_data_dir / train_file_path

    test_file_path = Path(params_yaml['process']['test_file'])
    test_data_path = processed_data_dir / test_file_path

    random_state = params_yaml['base']['random_state']
    target = [params_yaml['base']['target_col']]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    y_train = train[target]
    y_test = test[target]

    X_train = train.drop(target, axis = 1)
    X_test = test.drop(target, axis = 1)

    random_state = params_yaml['base']['random_state']
    n_est = params_yaml['train']['n_est']

    rf = RandomForestClassifier(n_estimators = n_est, random_state = random_state)
    rf.fit(X_train, y_train)

    model_dir = Path(params_yaml['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path('rf_model_1.pkl')
    with open(model_dir / model_file, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    param_yaml_path = 'D:/Projects/DVC_pipeline/params.yaml'
    training(param_yaml_path = param_yaml_path)

