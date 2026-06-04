# 下载apigen数据集

mkdir -p datasets/apigen
hf download Salesforce/xlam-function-calling-60k \
  xlam_function_calling_60k.json \
  --repo-type dataset \
  --local-dir datasets/apigen
python data/convert_function_calling_datasets.py \
  --datasets apigen \
  --base-dir datasets

# 下载toolace数据集

mkdir -p datasets/toolace
hf download Team-ACE/ToolACE \
  data.json \
  --repo-type dataset \
  --local-dir datasets/toolace
python data/convert_function_calling_datasets.py \
  --datasets toolace \
  --base-dir datasets

# 下载openr1_math数据集

python data/convert_rlvr_datasets.py \
  --datasets openr1_math \
  --base-dir datasets \
  --backend hf

# 下载scienceqa数据集

python data/convert_rlvr_datasets.py \
  --datasets scienceqa \
  --base-dir datasets \
  --backend hf

# 下载medmcqa数据集

python data/convert_rlvr_datasets.py \
  --datasets medmcqa \
  --base-dir datasets \
  --backend hf

# 下载taco数据集

python data/convert_rlvr_datasets.py \
  --datasets taco \
  --base-dir datasets \
  --backend hf
