import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from train import MetaLearner # Import model definition

def evaluate(model, test_loader, config, device):
    model.eval()
    all_accuracies_over_steps = []
    adaptive_results = []
    num_test_tasks = config['evaluation']['num_test_tasks']
    algorithm = config['experiment']['algorithm']
    num_inner_steps = config['training']['num_inner_steps']
    
    pbar = tqdm(test_loader, total=num_test_tasks, desc="Meta-Testing")

    for i, batch in enumerate(pbar):
        if i >= num_test_tasks:
            break
        support_x, support_y = batch['train']
        query_x, query_y = batch['test']
        support_x, support_y = support_x.squeeze(0).to(device), support_y.squeeze(0).to(device)
        query_x, query_y = query_x.squeeze(0).to(device), query_y.squeeze(0).to(device)

        with torch.no_grad():
            _, query_accuracies = model(support_x, support_y, query_x, query_y, is_train=False)
        
        if algorithm != 'protonet':
            all_accuracies_over_steps.append(query_accuracies)
            if algorithm == 'veloml':
                acc, steps = model.run_adaptive_inference(support_x, support_y, query_x, query_y)
                adaptive_results.append({'acc': acc, 'steps': steps})
        else:
            # ProtoNet accuracy is constant across "steps"
            all_accuracies_over_steps.append([query_accuracies] * (num_inner_steps + 1))

    results = {}
    if algorithm == 'protonet':
        mean_acc = np.mean([item[0] for item in all_accuracies_over_steps])
        print(f"ProtoNet Final Accuracy: {mean_acc:.4f} (+/- ...)")
        results['acc_curve'] = [mean_acc] * (num_inner_steps + 1)
        return results

    accuracies_df = pd.DataFrame(all_accuracies_over_steps)
    mean_accuracies = accuracies_df.mean(axis=0).values
    
    results['acc_curve'] = mean_accuracies
    
    if algorithm == 'veloml':
        adaptive_df = pd.DataFrame(adaptive_results)
        results['adaptive_acc'] = adaptive_df['acc'].mean()
        results['adaptive_steps_mean'] = adaptive_df['steps'].mean()
        results['adaptive_steps_dist'] = adaptive_df['steps'].values
        print(f"VeloML Adaptive Inference: Avg Acc = {results['adaptive_acc']:.4f}, Avg Steps = {results['adaptive_steps_mean']:.2f}")

    print(f"Final Accuracy (at step {num_inner_steps}): {mean_accuracies[-1]:.4f}")
    return results

def analyze_and_plot_results(all_results, config):
    print("\n--- Final Results Analysis ---")
    results_dir = config['experiment']['results_dir']
    images_dir = os.path.join(results_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    dataset = config['experiment']['dataset']
    n_way = config['experiment']['n_way']
    k_shot = config['experiment']['k_shot']
    dataset_name = f"{dataset.capitalize()} {n_way}-way {k_shot}-shot"
    num_inner_steps = config['training']['num_inner_steps']
    
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 6)})
    
    all_dfs = []
    for algorithm, results_per_seed in all_results.items():
        for seed_idx, result in enumerate(results_per_seed):
            df = pd.DataFrame({
                'step': range(len(result['acc_curve'])),
                'accuracy': result['acc_curve'],
                'algorithm': algorithm,
                'seed': seed_idx
            })
            all_dfs.append(df)
    
    if not all_dfs:
        print("No results to analyze.")
        return
        
    results_df = pd.concat(all_dfs, ignore_index=True)
    
    # Plot 1: Accuracy vs. Adaptation Steps
    plt.figure()
    sns.lineplot(data=results_df, x='step', y='accuracy', hue='algorithm', errorbar=('ci', 95))
    plt.title(f'Accuracy vs. Adaptation Steps ({dataset_name})')
    plt.xlabel('Adaptation Steps')
    plt.ylabel('Accuracy')
    plt.legend(title='Algorithm')
    plt.grid(True, which='both', linestyle='--')
    plot_path = os.path.join(images_dir, f'accuracy_vs_steps_{dataset}_{n_way}way_{k_shot}shot.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved accuracy plot to: {plot_path}")
    plt.close()

    # VeloML specific plots
    if 'veloml' in all_results:
        veloml_results = all_results['veloml']
        all_adaptive_steps = np.concatenate([res['adaptive_steps_dist'] for res in veloml_results if 'adaptive_steps_dist' in res])
        
        if len(all_adaptive_steps) > 0:
            plt.figure()
            sns.histplot(all_adaptive_steps, bins=np.arange(0, all_adaptive_steps.max() + 2) - 0.5, kde=False)
            plt.title(f'VeloML: Distribution of Stopping Steps ({dataset_name})')
            plt.xlabel('Number of Adaptation Steps')
            plt.ylabel('Frequency')
            plot_path = os.path.join(images_dir, f'stopping_steps_dist_veloml_{dataset}_{n_way}way_{k_shot}shot.pdf')
            plt.savefig(plot_path, format='pdf', bbox_inches='tight')
            print(f"Saved VeloML stopping steps plot to: {plot_path}")
            plt.close()

    # Final metrics reporting
    print("\n--- Performance Summary ---")
    summary_data = []
    for algorithm in results_df['algorithm'].unique():
        algo_df = results_df[results_df['algorithm'] == algorithm]
        final_acc_data = algo_df[algo_df['step'] == num_inner_steps]['accuracy']
        final_acc_mean = final_acc_data.mean()
        final_acc_ci = 1.96 * final_acc_data.std() / np.sqrt(len(final_acc_data))
        summary_data.append({
            'Algorithm': algorithm,
            'Final Accuracy': f"{final_acc_mean:.4f} ± {final_acc_ci:.4f}"
        })
        if algorithm == 'veloml' and 'veloml' in all_results:
            adaptive_accs = [res['adaptive_acc'] for res in all_results['veloml']]
            adaptive_steps = [res['adaptive_steps_mean'] for res in all_results['veloml']]
            summary_data.append({
                'Algorithm': 'VeloML (Adaptive)',
                'Final Accuracy': f"{np.mean(adaptive_accs):.4f} ± {1.96 * np.std(adaptive_accs) / np.sqrt(len(adaptive_accs)):.4f}"
            })
            print(f"VeloML Adaptive Avg Steps: {np.mean(adaptive_steps):.2f}")

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
