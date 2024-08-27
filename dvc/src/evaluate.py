import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import pickle

# from sklearn import processing
from matplotlib import pyplot as plt
from dvclive import Live
from sklearn import metrics

def evaluate(param_yaml_path, model, data, data_category, live):
    with open(param_yaml_path) as f:
        param_file = yaml.safe_load(f)

    target = [param_file['base']['target_col']]
    X = data.drop(target, axis = 1)
    y = data[target]

    predictions_by_class = model.predict_proba(X)
    y_pred = model.predict(X)
    predictions = predictions_by_class[:, 1]

    # Use DVClive to log
    avg_precision = metrics.precision_score(y, y_pred, average = 'macro')
    roc_auc = metrics.roc_auc_score(y, predictions_by_class, multi_class = 'ovr')

    if not live.summary:
        live.summary = {'avg_precision' : {}, 'roc_auc' : {}}
    live.summary['avg_precision'][data_category] = avg_precision
    live.summary['roc_auc'][data_category] = roc_auc

    # plots
    # live.log_sklearn_plot('roc', y, predictions_by_class, name = f'roc{data_category}')

    # precision, recall, pre_thresholds = metrics.precision_recall_curve(y, predictions)
    # nth_point = math.ceil(len(pre_thresholds) / 1000)
    # precision_points = list(zip(precision, recall, pre_thresholds))
    # precision_dir = Path(eval) / Path(prc)
    # Path.mkdir(precision_dir, exist_ok = True)
    # precision_file = Path(precision_dir) / f'{data-category}.json'
    
    # with opne(precision_file, 'w') as fd:
    #     json.dump(
    #         {
    #             'precision' : [
    #                 {'precision' : p, 'recall' : r, 'threshold' : t}
    #                 for p, r, t in precision_points
    #             ]
    #         },
    #         fd,
    #         indent = 4,
    #     )

    live.log_sklearn_plot('confusion_matrix', y,
                           predictions_by_class.argmax(-1),
                           name = f'cm/{data_category}')
    
    return ""

if __name__ == '__main__':
    param_yaml_path = 'D:/Projects/DVC_pipeline/params.yaml'
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)
    
    model_dir = Path(params_yaml['model_dir'])
    model_file = Path('rf_model_1.pkl')
    with open(model_dir / model_file, 'rb') as f:
        model = pickle.load(f)

    processed_data_dir = Path(params_yaml['process']['dir'])
    processed_train_data_path = Path(params_yaml['process']['train_file'])
    train_data_path = processed_data_dir / processed_train_data_path
    train = pd.read_csv(train_data_path)
    
    processed_test_data_path = Path(params_yaml['process']['test_file'])
    test_data_path = processed_data_dir / processed_test_data_path
    test = pd.read_csv(test_data_path)

    # Evaluate train and test datasets
    EVAL_PATH = 'eval'
    live = Live((Path(EVAL_PATH) / 'live'), dvcyaml = False, )
    evaluate(param_yaml_path, model, train, 'train', live)
    evaluate(param_yaml_path, model, test, 'test', live)
    live.make_summary()

    # Dump feature importances image
    fig, axes = plt.subplots(dpi = 100)
    fig.subplots_adjust(bottom = 0.2, top = 0.95)
    importances = model.feature_importances_
    X = train.drop(columns = 'species', axis = 1)
    feature_names = [f'feature {i}' for i in range(X.shape[1])]
    forest_importances = pd.Series(importances, index = feature_names).nlargest(n = 30)
    axes.set_ylabel('Mean decrease in impurity')
    forest_importances.plot.bar(ax = axes)

    fig.savefig(Path(EVAL_PATH) / 'importance.png')