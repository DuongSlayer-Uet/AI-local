# install vim
sudo apt update
sudo apt install vim

# git config
git remote set-url origin git@github.com:DuongSlayer-Uet/AI-local.git
git config --global user.email "buivanduong0705@gmail.com"
git config --global user.name "DuongSlayer"

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# httpx
pip install httpx

# uvicorn
pip install "uvicorn[standard]" fastapi
pip install python-multipart

# ollama setup
curl -fsSL https://ollama.com/install.sh | sh

# download qwen3:8b
ollama pull qwen3:8b

pip install --upgrade piper-tts

# transfpormers, datasets[audio], accelerate
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] accelerate

# mcp
pip install "mcp[cli]"

# vieneu
pip install vieneu
