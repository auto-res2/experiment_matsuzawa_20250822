import time
import uuid
import warnings
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
import torch
import gpytorch

try:
    from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient as Benchmark
except ImportError:
    print("HPOBench not found. Please install it with container support:")
    print("pip install hpobench[container]")
    print("And ensure you have Docker or Singularity installed and running.")
    exit()

warnings.filterwarnings("ignore", category=gpytorch.warnings.OldVersionWarning)

# --- Conceptual GPyTorch Model ---
class MultiFidelityGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_fidelities, ard_num_dims):
        super(MultiFidelityGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module_config = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_fidelities, rank=1)

    def forward(self, x):
        config_part = x[..., :-1]
        fidelity_part = x[..., -1].long()
        mean_x = self.mean_module(config_part)
        covar_config = self.covar_module_config(config_part)
        covar_fidelity = self.task_covar_module(fidelity_part)
        covar = covar_config * covar_fidelity
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

# --- HPO Algorithm Implementations ---
Trial = namedtuple('Trial', ['config', 'budget', 'config_id', 'rung_idx'])

class BaseOptimizer:
    def __init__(self, config_space, seed):
        self.config_space = config_space
        self.config_space.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.history = []
        self.incumbent_trajectory = pd.DataFrame(columns=["wall_clock_time", "best_found_value"])
        self.incumbent_value = float('inf')
        self.start_time = time.time()
        self.min_budget, self.max_budget, self.eta = 1, 10, 3
        budgets = [self.max_budget / (self.eta**i) for i in range(4)][::-1]
        self.budgets = [int(b) for b in budgets if b >= self.min_budget]
        self.num_rungs = len(self.budgets)
        self.rungs = {i: [] for i in range(self.num_rungs)}
        self.pending_trials = {}

    def ask(self, num_configs):
        raise NotImplementedError

    def tell(self, results):
        for res in results:
            trial = res['trial']
            value = res['value']
            if trial.config_id in self.pending_trials:
                del self.pending_trials[trial.config_id]
            result_entry = {'config_id': trial.config_id, 'config': trial.config, 'budget': trial.budget, 'value': value}
            self.history.append(result_entry)
            self.rungs[trial.rung_idx].append(result_entry)
            if value < self.incumbent_value:
                self.incumbent_value = value
                new_entry = pd.DataFrame([{"wall_clock_time": time.time() - self.start_time, "best_found_value": self.incumbent_value}])
                self.incumbent_trajectory = pd.concat([self.incumbent_trajectory, new_entry], ignore_index=True)
    
    def get_incumbent_trajectory(self):
        return self.incumbent_trajectory

    def _get_new_config(self):
        return self.config_space.sample_configuration()

class RandomSearch(BaseOptimizer):
    def ask(self, num_configs):
        time.sleep(0.001)
        trials_to_evaluate = []
        for _ in range(num_configs):
            if len(self.pending_trials) >= num_configs: break
            config = self._get_new_config()
            config_id = str(uuid.uuid4())
            trial = Trial(config, self.budgets[0], config_id, 0)
            self.pending_trials[config_id] = trial
            trials_to_evaluate.append(trial)
        return trials_to_evaluate

class ASHA(BaseOptimizer):
    def ask(self, num_configs):
        time.sleep(0.002)
        trials_to_evaluate = []
        promotable = self._get_promotable_configs()
        for config_id, old_rung_idx, config in promotable:
            if len(trials_to_evaluate) >= num_configs: break
            if config_id not in self.pending_trials:
                new_rung_idx = old_rung_idx + 1
                trial = Trial(config, self.budgets[new_rung_idx], config_id, new_rung_idx)
                self.pending_trials[config_id] = trial
                trials_to_evaluate.append(trial)
        while len(trials_to_evaluate) < num_configs:
            config = self._get_new_config()
            config_id = str(uuid.uuid4())
            trial = Trial(config, self.budgets[0], config_id, 0)
            self.pending_trials[config_id] = trial
            trials_to_evaluate.append(trial)
        return trials_to_evaluate

    def _get_promotable_configs(self):
        promotions = []
        for i in range(self.num_rungs - 1):
            rung_results = self.rungs[i]
            if not rung_results: continue
            n_to_promote = len(rung_results) // self.eta
            if n_to_promote == 0: continue
            sorted_rung = sorted(rung_results, key=lambda x: x['value'])
            for res in sorted_rung[:n_to_promote]:
                is_promoted = any(h['config_id'] == res['config_id'] and h['budget'] > res['budget'] for h in self.history)
                if not is_promoted:
                    promotions.append((res['config_id'], i, res['config']))
        return promotions

class BOHB(ASHA):
    def _get_new_config(self):
        time.sleep(0.02)
        if self.rng.rand() < 0.5 or not self.history:
            return self.config_space.sample_configuration()
        else:
            good_configs = sorted(self.history, key=lambda x: x['value'])[:max(1, len(self.history)//10)]
            best = self.rng.choice(good_configs)['config']
            new_config = self.config_space.sample_configuration()
            for hp in new_config:
                if self.rng.rand() < 0.7 and hp in best:
                    new_config[hp] = best[hp]
            return new_config

class CAR_HPO(BOHB):
    def __init__(self, config_space, seed):
        super().__init__(config_space, seed)
        self.n_new_counter = defaultdict(int)
        self.n_new_trigger = 4

    def tell(self, results):
        super().tell(results)
        for res in results:
            self.n_new_counter[res['trial'].rung_idx] += 1
            if self.n_new_counter[res['trial'].rung_idx] >= self.n_new_trigger:
                time.sleep(0.08)
                self.n_new_counter[res['trial'].rung_idx] = 0
    
    def _get_promotable_configs(self):
        promotions = []
        for i in range(self.num_rungs - 1):
            rung_i_results = self.rungs[i]
            rung_i_plus_1_results = self.rungs[i+1]
            if not rung_i_results: continue
            threshold = np.median([r['value'] for r in rung_i_plus_1_results]) if rung_i_plus_1_results else float('inf')
            for res in rung_i_results:
                is_promoted = any(h['config_id'] == res['config_id'] and h['budget'] > res['budget'] for h in self.history)
                if not is_promoted and res['value'] < threshold:
                    promotions.append((res['config_id'], i, res['config']))
        return promotions

class DyHPO(CAR_HPO):
    def tell(self, results):
        time.sleep(0.05 * len(results))
        super(BOHB, self).tell(results)

# --- Experiment Runner ---
def run_hpo_trial(optimizer_class, benchmark_name, n_workers, wall_clock_limit_secs, seed):
    try:
        b = Benchmark(benchmark_name=benchmark_name, rng=seed)
    except Exception as e:
        print(f"ERROR: Could not initialize benchmark '{benchmark_name}'. Is the container engine running? Details: {e}")
        return pd.DataFrame()
    
    config_space = b.get_configuration_space(seed=seed)
    optimizer = optimizer_class(config_space=config_space, seed=seed)

    trajectory = []
    start_time = time.time()
    total_scheduler_overhead = 0.0
    
    while time.time() - start_time < wall_clock_limit_secs:
        ask_start_time = time.time()
        trials_to_evaluate = optimizer.ask(num_configs=n_workers)
        total_scheduler_overhead += time.time() - ask_start_time
        if not trials_to_evaluate: break

        results = []
        for trial in trials_to_evaluate:
            eval_result = b.objective_function(configuration=trial.config, budget=trial.budget)
            results.append({'trial': trial, 'value': eval_result['function_value']})
        
        tell_start_time = time.time()
        optimizer.tell(results)
        total_scheduler_overhead += time.time() - tell_start_time
        
        incumbent_traj_df = optimizer.get_incumbent_trajectory()
        if not incumbent_traj_df.empty:
            current_best = incumbent_traj_df['best_found_value'].iloc[-1]
            elapsed_time = time.time() - start_time
            if not trajectory or trajectory[-1]['wall_clock_time'] < elapsed_time:
                trajectory.append({'wall_clock_time': elapsed_time, 'best_found_value': current_best, 'scheduler_overhead': total_scheduler_overhead})
    
    if trajectory:
        final_entry = trajectory[-1].copy()
        final_entry['wall_clock_time'] = wall_clock_limit_secs
        trajectory.append(final_entry)
            
    return pd.DataFrame(trajectory)
