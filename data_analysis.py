import os
import json
import pandas as pd


dataset_name = 'archi'
dataset_dir = f"results-ordered/{dataset_name}"
columns = ['task_type', 'node_cls_label', 'edge_removal', 'type_semantic_removal', 'distance', 'cleanse', 'ordered', 'use_node_types', 'use_edge_types']
metrics = ['eval_accuracy', 'eval_macro_precision', 'eval_macro_recall', 'eval_macro_f1']
header = ['Config', 'Task Type', 'Target', 'Edge Fraction', 'Percentage Masked', 'Hops', 'Cleansed', 'Ordered', 'Use Node Types', 'Use Edge Types'] + ["Accuracy", "Precision", "Recall", "F1"]

column_signs = {
    "Edge Fraction": 0,
    'Percentage Masked': 0,
    "Hops": 1,
    "Cleansed": 1,
    "Ordered": 1,
    "Use Node Types": 1,
    "Use Edge Types": 1,
    "Task Type": 1
}
rows = list()

for config in os.listdir(dataset_dir):
    best_result = None
    run_config = None
    if os.path.exists(os.path.join(dataset_dir, config, "trainer_state.json")):
        with open(os.path.join(dataset_dir, config, "trainer_state.json")) as f:
            trainer_state = json.load(f)
            # print(os.path.join(dataset_dir, config, "trainer_state.json"))
            best_global_step = trainer_state['best_global_step']
            best_result = {r['step']: r for r in trainer_state['log_history'] if 'eval_accuracy' in r}[best_global_step]

    if os.path.exists(os.path.join(dataset_dir, config, "run_config.json")):
        with open(os.path.join(dataset_dir, config, "run_config.json")) as f:
            run_config = json.load(f)
    

    if best_result and run_config and not run_config['extraction_config']['use_node_types']:
        row = [run_config[k]  if k in run_config else "" for k in columns] + [best_result[k] for k in metrics]
        row = [config] + [int(v) if isinstance(v, bool) else v for v in row]
        rows.append(row)
        
df = pd.DataFrame(rows, columns=header)
df.to_excel(f'comparison-results/{dataset_name}-comparative-ordered.xlsx', index=False)
df.to_csv(f'comparison-results/{dataset_name}-comparative-ordered.csv', index=False)


def get_sign(score, column):
    column_sign = column_signs[column]
    if column_sign:
        return score > 0
    return score < 0
    

for task_type in df['Task Type'].unique():
    target_comparisons = dict()
    for target_name in df['Target'].unique():
        target_df = df.loc[(df["Target"] == target_name) & (df["Task Type"] == task_type)]
        config_columns = ['Task Type', 'Edge Fraction', 'Percentage Masked', 'Hops', 'Cleansed', 'Ordered', 'Use Node Types', 'Use Edge Types']
        column_results = dict()
        for column in config_columns:
            if column == 'Task Type':
                continue
            column_configs = dict()
            for i, row in target_df.iterrows():
                row_id = " | ".join([f"{k}:{v}" for k, v in {c: row[c] for c in config_columns if c != column}.items()])

                if row_id not in column_configs:
                    column_configs[row_id] = list()
                column_configs[row_id].append({r: row[r] for r in [column, "Config", "Accuracy", "Precision", "Recall", "F1"]})
            
            for row_id in column_configs:
                rows = column_configs[row_id]
                sorted_rows = sorted(rows, key=lambda x: x[column])
                scores = {k: list() for k in sorted_rows[0].keys() if k != column}
                for source, target in zip(sorted_rows[:-1], sorted_rows[1:]):
                    for k in source:
                        if k not in [column, "Config"]:
                            scores[k].append(target[k] - source[k])
                            
                column_configs[row_id].append(scores)
            
            metrics_to_consider = list(column_configs.values())[0][-1].keys()
            delta_scores = dict()
            for metric_to_consider in metrics_to_consider:
                
                column_delta = [1 if get_sign(delta, column) else 0 for value in list(column_configs.values()) for delta in value[-1][metric_to_consider]]
                if len(column_delta) == 0:
                    continue
                column_delta_str = f"{sum(column_delta)/len(column_delta)}% | {sum(column_delta)}/{len(column_delta)}"
                delta_scores[metric_to_consider] = column_delta_str
            
            column_results[column] = {
                'Configs': column_configs,
                "Delta": delta_scores
            }
            
        target_comparisons[target_name] = column_results
        
            
    with open(f'comparison-results/{dataset_name}-comparative-{task_type}-ordered.json', 'w') as f:
        json.dump(target_comparisons, f, indent=4)