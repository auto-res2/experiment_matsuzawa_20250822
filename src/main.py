import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from spikingjelly.activation_based import neuron, layer, functional, learning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import argparse

# It's assumed this script is run from the project root, e.g., `python src/main.py`
from preprocess import get_cifar10_data
from train import train_one_epoch
from evaluate import evaluate

# --- 1. Core FG-OTTT Implementation ---
class SparseOTTTFunction(Function):
    @staticmethod
    def forward(ctx, x_seq, weight, pre_act_trace, config, is_conv=False, conv_params=None):
        T, batch_size = x_seq.shape[0], x_seq.shape[1]
        x_flat = x_seq.flatten(0, 1)

        if is_conv:
            output_flat = F.conv2d(x_flat, weight, **conv_params)
            output_shape = (T, batch_size, output_flat.shape[1], output_flat.shape[2], output_flat.shape[3])
            output_seq = output_flat.view(*output_shape)
        else:
            output_flat = F.linear(x_flat, weight)
            output_shape = (T, batch_size, output_flat.shape[1])
            output_seq = output_flat.view(*output_shape)

        with torch.no_grad():
            post_act_trace_approx = torch.mean((output_seq > 0).float(), dim=0)
            if is_conv:
                hebbian_mask = torch.rand_like(weight) < (config['s_target'] * 0.9)
            else:
                hebbian_mask = (pre_act_trace.unsqueeze(0) > 0.1) & (post_act_trace_approx.mean(dim=0).unsqueeze(1) > 0.1)
            
            stochastic_mask = torch.rand_like(weight) < (config['s_target'] * 0.1)
            mask = hebbian_mask | stochastic_mask
            
            current_density = mask.float().mean()
            if current_density > config['s_target'] and config['s_target'] < 1.0:
                k = int(config['s_target'] * mask.numel())
                flat_mask = mask.view(-1)
                indices = torch.where(flat_mask)[0]
                perm = torch.randperm(indices.size(0), device=mask.device)
                selected_indices = indices[perm[:k]]
                new_flat_mask = torch.zeros_like(flat_mask, dtype=torch.bool)
                new_flat_mask[selected_indices] = True
                mask = new_flat_mask.view(mask.shape)

        ctx.save_for_backward(x_flat, weight, mask)
        ctx.is_conv = is_conv
        ctx.conv_params = conv_params
        ctx.input_shape = x_flat.shape
        ctx.T = T

        return output_seq, mask.float().mean()

    @staticmethod
    def backward(ctx, grad_output, grad_density):
        input_flat, weight, mask = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_output_flat = grad_output.flatten(0, 1)

        if ctx.needs_input_grad[1]:
            if ctx.is_conv:
                grad_weight = torch.nn.grad.conv2d_weight(input_flat, weight.shape, grad_output_flat, **ctx.conv_params)
            else:
                grad_weight = grad_output_flat.t().matmul(input_flat)
            grad_weight.mul_(mask)

        if ctx.needs_input_grad[0]:
            sparse_weights = weight * mask
            if ctx.is_conv:
                grad_input_flat = torch.nn.grad.conv2d_input(ctx.input_shape, sparse_weights, grad_output_flat, **ctx.conv_params)
            else:
                grad_input_flat = grad_output_flat.matmul(sparse_weights)
            grad_input = grad_input_flat.view(ctx.T, -1, *input_flat.shape[1:])

        return grad_input, grad_weight, None, None, None, None

class SparseLayer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.layer = layer
        self.config = config
        self.is_conv = isinstance(self.layer, nn.Conv2d)
        trace_shape = self.layer.in_features if not self.is_conv else self.layer.in_channels
        self.register_buffer('pre_act_trace', torch.zeros(trace_shape))

    def forward(self, x_seq):
        with torch.no_grad():
            current_act = x_seq.mean(dim=[0, 1])
            if self.is_conv:
                current_act = current_act.mean(dim=[1, 2])
            self.pre_act_trace.mul_(0.9).add_(current_act, alpha=0.1)
        
        conv_params = {}
        if self.is_conv:
            conv_params = {'stride': self.layer.stride, 'padding': self.layer.padding, 'dilation': self.layer.dilation, 'groups': self.layer.groups}
        
        output_seq, density = SparseOTTTFunction.apply(x_seq, self.layer.weight, self.pre_act_trace, self.config, self.is_conv, conv_params)
        
        if self.layer.bias is not None:
            bias_shape = [1, 1, -1] + ([1, 1] if self.is_conv else [])
            output_seq += self.layer.bias.view(*bias_shape)

        return output_seq, density

# --- 2. Model Architectures ---
class VGG_SNN(nn.Module):
    def __init__(self, num_classes=10, model_type='BPTT', s_target=0.1):
        super().__init__()
        self.model_type = model_type
        self.s_target = s_target
        self.densities = []

        conv_params = {'kernel_size': 3, 'padding': 1}
        pool = layer.AvgPool2d(kernel_size=2)
        lif = neuron.LIFNode(tau=2.0, surrogate_function=learning.ATan(), detach_reset=True)
        
        self.net = nn.ModuleList([
            self._create_layer(3, 128, is_conv=True, conv_params=conv_params), lif, pool,
            self._create_layer(128, 256, is_conv=True, conv_params=conv_params), lif, pool,
            self._create_layer(256, 512, is_conv=True, conv_params=conv_params), lif, pool,
            layer.Flatten(),
            self._create_layer(512 * 4 * 4, 1024, is_conv=False), lif,
            self._create_layer(1024, num_classes, is_conv=False), lif
        ])

    def _create_layer(self, in_c, out_c, is_conv, conv_params={}):
        if self.model_type == 'FG-OTTT':
            l = nn.Conv2d(in_c, out_c, **conv_params) if is_conv else nn.Linear(in_c, out_c)
            return SparseLayer(l, {'s_target': self.s_target})
        else:
            if is_conv:
                return layer.Conv2d(in_c, out_c, **conv_params)
            else:
                return layer.Linear(in_c, out_c)

    def forward(self, x_seq):
        self.densities = []
        for module in self.net:
            if isinstance(module, SparseLayer):
                x_seq, density = module(x_seq)
                self.densities.append(density)
            else:
                x_seq = module(x_seq)
        functional.reset_net(self)
        return x_seq

    def get_avg_density(self):
        if not self.densities: return 1.0
        return torch.tensor(self.densities).mean().item()

# --- 3. Plotting ---
def plot_results(results_df, image_dir):
    print("\n--- Generating Plots ---")
    os.makedirs(image_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Epoch', y='Train Loss', hue='Method')
    plt.title('Training Loss vs. Epochs'); plt.grid(True)
    plt.savefig(os.path.join(image_dir, "training_loss_comparison.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Epoch', y='Test Accuracy', hue='Method')
    plt.title('Test Accuracy vs. Epochs'); plt.grid(True)
    plt.savefig(os.path.join(image_dir, "accuracy_comparison.pdf"), bbox_inches="tight")
    plt.close()

    final_results = results_df.loc[results_df.groupby('Method')['Epoch'].idxmax()]
    final_results['Total Train Time (min)'] = final_results.groupby('Method')['Train Time (s)'].transform('sum') / 60
    final_results['Avg Bwd GFLOPs'] = results_df.groupby('Method')['Bwd GFLOPs'].transform('mean')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_results, x='Total Train Time (min)', y='Test Accuracy', hue='Method', s=200, style='Method')
    plt.title('Accuracy vs. Total Training Time'); plt.xscale('log'); plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(os.path.join(image_dir, "accuracy_vs_time.pdf"), bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_results, x='Avg Bwd GFLOPs', y='Test Accuracy', hue='Method', s=200, style='Method')
    plt.title('Accuracy vs. Backward Pass Computation'); plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(os.path.join(image_dir, "accuracy_vs_bwd_gflops.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {image_dir}")

# --- 4. Main Experiment Driver ---
def run_experiment(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = ".research/iteration1"
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_loader, test_loader = get_cifar10_data(args.batch_size, args.T, data_dir='data')

    experiment_configs = [
        {'model_type': 'Dense-OTTT', 's_target': 1.0},
        {'model_type': 'BPTT', 's_target': 1.0},
        {'model_type': 'HS-OTTT', 's_target': 0.2},
        {'model_type': 'FG-OTTT', 's_target': 0.2},
        {'model_type': 'FG-OTTT', 's_target': 0.1},
    ]
    all_results = []

    for config in experiment_configs:
        for seed in range(args.num_seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            method_name = f"{config['model_type']}_{config['s_target']}" if config['s_target'] < 1.0 else config['model_type']
            print(f"\n--- Running: {method_name} | Seed: {seed + 1}/{args.num_seeds} ---")

            model = VGG_SNN(model_type=config['model_type'], s_target=config['s_target']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            for epoch in range(1, args.epochs + 1):
                train_config = {**config, 'epoch': epoch, 'lambda_sparsity': 0.1 * min(1.0, epoch / (args.epochs * 0.1))}

                loss, acc, epoch_time, density, bwd_gflops = train_one_epoch(model, train_loader, optimizer, scheduler, device, train_config)
                test_acc = evaluate(model, test_loader, device)
                
                print(f"Epoch {epoch}: Loss={loss:.4f}, Train Acc={acc:.4f}, Test Acc={test_acc:.4f}, Time={epoch_time:.2f}s, Density={density:.4f}")
                
                all_results.append({'Method': method_name, 'Seed': seed, 'Epoch': epoch, 'Train Loss': loss, 'Train Accuracy': acc, 'Test Accuracy': test_acc, 'Train Time (s)': epoch_time, 'Gradient Density': density, 'Bwd GFLOPs': bwd_gflops})

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_dir, "experiment_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nFull experiment results saved to {results_csv_path}")
    plot_results(results_df, image_dir)

# --- 5. Test Function ---
def test_function():
    print("\n--- Running Quick Test Function ---")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    T = 4
    dummy_loader = [(torch.rand(T, 4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(2)]
    configs = [{'model_type': 'Dense-OTTT', 's_target': 1.0}, {'model_type': 'HS-OTTT', 's_target': 0.2}, {'model_type': 'FG-OTTT', 's_target': 0.2}, {'model_type': 'BPTT', 's_target': 1.0}]

    for config in configs:
        print(f"Testing model: {config['model_type']}")
        model = VGG_SNN(model_type=config['model_type'], s_target=config['s_target']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        train_cfg = {**config, 'epoch': 1, 'lambda_sparsity': 0.1}
        train_one_epoch(model, dummy_loader, optimizer, scheduler, device, train_cfg)
        evaluate(model, dummy_loader, device)
        print(f"Test for {config['model_type']} PASSED.")
    print("--- Quick Test Function Finished Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FG-OTTT Benchmark Experiment')
    parser.add_argument('--run_test', action='store_true', help='Run the quick test function.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--T', type=int, default=16, help='Number of simulation time steps.')
    parser.add_argument('--num_seeds', type=int, default=2, help='Number of random seeds.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID.')
    args = parser.parse_args()

    if args.run_test:
        test_function()
    else:
        run_experiment(args)
