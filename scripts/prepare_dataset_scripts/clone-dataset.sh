"""
  Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    ./clone-dataset.sh

""" 

# 定义变量
MAX_RETRIES=500
RETRY_COUNT=0

# export HF_HUB_ENABLE_HF_TRANSFER=1
# 开始重试循环

export HF_ENDPOINT=https://hf-mirror.com
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  echo "Attempt $((RETRY_COUNT + 1))"
  
  # 运行Python脚本
  huggingface-cli download --token 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  --repo-type dataset --resume-download allenai/objaverse --local-dir objaverse/ 
  
  # 检查退出状态码
  if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
    exit 0
  else
    echo "Python script failed. Retrying..."
    RETRY_COUNT=$((RETRY_COUNT + 1))
  fi
done