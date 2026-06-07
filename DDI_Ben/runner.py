"""
Automated runner for DDI models on Kaggle
Simply enable/disable models in config.yaml and run!
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_args(args_dict):
    """Convert arguments dictionary to command line arguments"""
    cmd_args = []
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key}")
        else:
            cmd_args.extend([f"--{key}", str(value)])
    return cmd_args

def run_ddi_ben_model(model_name, model_config, base_dir):
    """Run DDI_Ben model"""
    print(f"\n{'='*80}")
    print(f"Starting {model_name} - {model_config['wandb_name']}")
    print(f"{'='*80}\n")
    
    script_path = os.path.join(base_dir, "DDI_Ben", "main.py")
    cmd = [sys.executable, script_path, 
           "--model", model_name.replace("_TWOSIDES", ""),
           "--name", model_config['wandb_name']]
    cmd.extend(build_args(model_config['args']))
    
    result = subprocess.run(cmd, cwd=os.path.join(base_dir, "DDI_Ben"))
    return result.returncode == 0

def run_emergnn_model(model_name, model_config, base_dir, dataset_type):
    """Run EmerGNN model"""
    print(f"\n{'='*80}")
    print(f"Starting {model_name} - {model_config['wandb_name']}")
    print(f"{'='*80}\n")
    
    folder = "DrugBank" if dataset_type == "emergnn_drugbank" else "TWOSIDES"
    script_path = os.path.join(base_dir, "EmerGNN", folder, "evaluate.py")
    cmd = [sys.executable, script_path]
    cmd.extend(build_args(model_config['args']))
    
    result = subprocess.run(cmd, cwd=os.path.join(base_dir, "EmerGNN", folder))
    return result.returncode == 0

def main():
    """Main runner function"""
    # Load config
    base_dir = Path(__file__).parent.absolute()
    config = load_config(os.path.join(base_dir, 'config.yaml'))
    
    # Set wandb key
    wandb_config = config.get('wandb', {})
    if 'wandb_key' in wandb_config:
        os.environ['WANDB_API_KEY'] = wandb_config['wandb_key']
    
    # Get enabled models
    models_config = config.get('models', {})
    enabled_models = {k: v for k, v in models_config.items() if v.get('enabled', False)}
    
    if not enabled_models:
        print("⚠️  No models enabled in config.yaml!")
        print("Edit config.yaml and set 'enabled: true' for models you want to run.")
        return
    
    print(f"\n{'='*80}")
    print(f"🚀 DDI Models Auto-Runner for Kaggle")
    print(f"{'='*80}")
    print(f"Enabled models: {', '.join(enabled_models.keys())}")
    print(f"Total: {len(enabled_models)} models")
    print(f"{'='*80}\n")
    
    # Run models
    results = {}
    for model_name, model_config in enabled_models.items():
        model_type = model_config.get('type', 'ddi_ben')
        
        try:
            if model_type == 'ddi_ben':
                success = run_ddi_ben_model(model_name, model_config, base_dir)
            elif model_type in ['emergnn_drugbank', 'emergnn_twosides']:
                success = run_emergnn_model(model_name, model_config, base_dir, model_type)
            else:
                print(f"❌ Unknown model type: {model_type}")
                success = False
            
            results[model_name] = "✅ Success" if success else "❌ Failed"
        except Exception as e:
            print(f"\n❌ Error running {model_name}: {e}")
            results[model_name] = f"❌ Error"
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    for model_name, result in results.items():
        print(f"{model_name:30} {result}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
