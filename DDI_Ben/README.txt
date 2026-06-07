================================================================================
DDI MODELS AUTO-RUNNER FOR KAGGLE
================================================================================

🚀 QUICK START
===============================================================================

CÁCH NHANH NHẤT: Dùng Notebook
-------------------------------
1. Mở kaggle_runner.ipynb
2. Chạy cell 1: Cài PyYAML
3. Edit cell 2: Bật/tắt models (%%writefile config.yaml)
4. Chạy cell 3: !python runner.py

DONE! 🎉

Hoặc qua Terminal:
------------------
1. !pip install PyYAML
2. Edit config.yaml (bật/tắt models)
3. !python runner.py

===============================================================================
📁 FILES
===============================================================================

config.yaml          - Cấu hình chính (edit file này)
config_test.yaml     - Config test (5 epochs) để thử nghiệm
runner.py           - Script tự động chạy models
kaggle_runner.ipynb - Notebook template cho Kaggle
requirements.txt    - Dependencies (đã có PyYAML)

===============================================================================
⚙️ CONFIGURATION
===============================================================================

Sửa config.yaml:

models:
  MLP:
    enabled: true          ← Bật/tắt ở đây
    wandb_name: "MLP"      ← Tên trên wandb
    args:
      dataset: "drugbank"  ← Tùy chỉnh hyperparameters
      lr: 0.0003
      epoch: 100
      ...

===============================================================================
🎯 MODELS HỖ TRỢ
===============================================================================

DDI_Ben models:
  ✓ MLP
  ✓ Decagon
  ✓ TIGER
  ✓ SSI-DDI
  ✓ MRCGNN
  ✓ KGE

EmerGNN models:
  ✓ EmerGNN_DrugBank
  ✓ EmerGNN_TWOSIDES

===============================================================================
📊 METRICS
===============================================================================

DrugBank:  Accuracy, Macro F1, Kappa
TWOSIDES:  ROC-AUC, PR-AUC, AP@K

Tất cả được log tự động lên Wandb!

===============================================================================
💡 EXAMPLES
===============================================================================

# Chạy tất cả models đã enabled
!python runner.py

# Chạy test nhanh (5 epochs)
!python runner.py

# Edit trong Python
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Chỉ chạy MLP
for model in config['models']:
    config['models'][model]['enabled'] = (model == 'MLP')

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

!python runner.py

===============================================================================
🔧 WANDB SETUP
===============================================================================

Option 1: Set trong config.yaml
  wandb:
    wandb_key: "your-api-key"

Option 2: Kaggle Secrets
  Thêm WANDB_API_KEY vào Kaggle Secrets

Option 3: Environment
  import os
  os.environ['WANDB_API_KEY'] = 'your-key'

===============================================================================

Để chạy, chỉ cần:
1. Edit config.yaml (bật/tắt models)
2. !python runner.py

Xem wandb để theo dõi training!

===============================================================================
