import os
import time
import argparse
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.preprocess import get_datasets, PoisonedDataset
from src.preprocess import BadNetsAttack, BlendedAttack, WaNetAttack, SinusoidalAttack
from src.train import get_model, VanillaTrainer, UltraCPlusTrainer, EpicTrainer, FinePuningTrainer
from src.train import FUNCTORCH_AVAILABLE
from src.evaluate import plot_training_curves, save_results_summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_experiment(config):
    """
    Main function to run the backdoor defense experiment based on a config file.
    """
    set_seed(config['seed'])
    
    # Create output directories
    images_dir = os.path.join(config['output_dir'], 'images')
    models_dir = os.path.join(config['output_dir'], 'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading dataset: {config['dataset']}...")
    train_dataset_clean, test_dataset, num_classes = get_datasets(config['dataset'])

    # 2. Setup Attack
    print(f"Setting up attack: {config['attack']['name']}...")
    attack_name = config['attack']['name']
    attack_params = config['attack']['params']
    if attack_name == 'BadNets': attack = BadNetsAttack(**attack_params)
    elif attack_name == 'Blended': attack = BlendedAttack(**attack_params)
    elif attack_name == 'WaNet': attack = WaNetAttack(**attack_params)
    elif attack_name == 'Sinusoidal': attack = SinusoidalAttack(**attack_params)
    else: raise ValueError(f"Unknown attack: {attack_name}")

    # 3. Poison Dataset and create DataLoaders
    poisoned_train_dataset = PoisonedDataset(
        dataset=train_dataset_clean, 
        attack=attack,
        poison_rate=config['attack']['poison_rate']
    )
    train_loader = DataLoader(poisoned_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    all_results = {}

    # 4. Run for each defense
    for defense in config['defenses']:
        print(f"\n{'='*25} RUNNING DEFENSE: {defense} {'='*25}")
        
        # Skip defenses with missing dependencies
        if defense in ['ULTRA-C+', 'EPIC'] and not FUNCTORCH_AVAILABLE:
            print(f"Skipping {defense}: dependency 'torch.func' not available.")
            continue

        model = get_model(config['model'], num_classes)
        trainer = None
        attack_config_for_trainer = {'name': attack_name, 'attack_instance': attack}
        start_time = time.time()

        if defense == 'Vanilla':
            trainer = VanillaTrainer(model, train_loader, test_loader, num_classes, defense, DEVICE)
        elif defense == 'ULTRA-C+':
            trainer = UltraCPlusTrainer(model, train_loader, test_loader, num_classes, defense, DEVICE)
        elif defense == 'EPIC':
            trainer = EpicTrainer(model, train_loader, test_loader, num_classes, defense, DEVICE)
        elif defense == 'FP':
            trainer = FinePuningTrainer(model, train_loader, test_loader, num_classes, defense, DEVICE)
        
        if trainer:
            trainer.train(config['epochs'], config['lr'], attack_config_for_trainer)
            
            end_time = time.time()
            training_time = (end_time - start_time)
            
            # Final evaluation
            final_ca, final_asr = trainer.evaluate(attack)
            
            # Save the trained model
            model_path = os.path.join(models_dir, f"{config['model']}_{defense}_{attack_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved trained model to {model_path}")

            print(f"\n{trainer.get_log_prefix()} [SUMMARY]")
            print(f"  Final Clean Accuracy (CA): {final_ca:.2f}%")
            print(f"  Final Attack Success Rate (ASR): {final_asr:.2f}%")
            print(f"  Total Training Time: {training_time/60:.2f} minutes")

            all_results[defense] = {
                'CA': final_ca, 'ASR': final_asr,
                'Training Time (s)': training_time,
                'Dataset': config['dataset'], 'Model': config['model'], 'Attack': attack_name
            }
            
            condition = f"{config['dataset']}_{config['model']}_{attack_name}"
            plot_training_curves(trainer.history, defense, condition, images_dir)
        else:
             print(f"Defense {defense} is not implemented or was skipped.")

    # 5. Save and plot summary
    if all_results:
        condition = f"{config['dataset']}_{config['model']}_{attack_name}"
        save_results_summary(all_results, condition, images_dir)

    print("\nExperiment finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Backdoor Defense Experiments.')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    print("Starting experiment with the following configuration:")
    print(json.dumps(config, indent=2))
    run_experiment(config)
