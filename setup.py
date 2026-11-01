#!/usr/bin/env bash
# setup_server.sh
# Setup server cho pipeline voice assistant: Ollama (qwen3:8b) + faster-whisper (CUDA)
set -euo pipefail

echo "==> [1/7] Dọn dẹp & cập nhật apt..."
sudo apt clean
sudo apt update
sudo apt install -y zstd

echo "==> [3/7] Cài Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama --version

echo "==> [4/7] Tải model qwen3:8b (có thể mất một lúc)..."
ollama pull qwen3:8b
# Test nhanh sau khi setup xong: ollama run qwen3:8b

echo "==> [5/7] Tạo Python venv cho whisper-test..."
mkdir -p ~/whisper-test
cd ~/whisper-test
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip
pip install faster-whisper
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

echo "==> [6/6] Thiết lập LD_LIBRARY_PATH cho cuBLAS/cuDNN..."
CUDA_LIB_PATH=$(python -c 'import os, nvidia.cublas, nvidia.cudnn; print(os.path.dirname(nvidia.cublas.__path__[0]) + "/cublas/lib:" + os.path.dirname(nvidia.cudnn.__path__[0]) + "/cudnn/lib")')
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"

# Ghi vào ~/.bashrc để không phải export lại mỗi lần mở terminal mới
if ! grep -qF "${CUDA_LIB_PATH}" ~/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "# faster-whisper: cuBLAS/cuDNN libs"
    echo "export LD_LIBRARY_PATH=\"${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH:-}\""
  } >> ~/.bashrc
  echo "   -> Đã thêm LD_LIBRARY_PATH vào ~/.bashrc"
else
  echo "   -> LD_LIBRARY_PATH đã có sẵn trong ~/.bashrc, bỏ qua"
fi

echo ""
echo "==> Hoàn tất setup server!"
echo "   - Chạy 'source ~/.bashrc' (hoặc mở terminal mới) để áp dụng LD_LIBRARY_PATH."
echo "   - venv nằm ở ~/whisper-test/.venv (activate bằng: source ~/whisper-test/.venv/bin/activate)"
echo "   - Test Ollama: ollama run qwen3:8b"
