import xgboost as xgb
import torch
import numpy as np

print(f'XGBoost version: {xgb.__version__}')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')

# Test XGBoost GPU
dtrain = xgb.DMatrix(np.array([[1,2],[3,4]]), label=[1,0])
params = {'device': 'cuda', 'tree_method': 'gpu_hist', 'objective': 'binary:logistic'}

try:
    model = xgb.train(params, dtrain, num_boost_round=5)
    print('✓ XGBoost GPU FUNZIONA!')
except Exception as e:
    print(f'✗ Errore GPU: {e}')
