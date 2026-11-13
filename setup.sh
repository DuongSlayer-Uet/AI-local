# install vim
sudo apt update
sudo apt install vim

# git config
git remote set-url origin git@github.com:DuongSlayer-Uet/AI-local.git
git config --global user.email "buivanduong0705@gmail.com"
git config --global user.name "DuongSlayer"

# git clone
git clone https://github.com/DuongSlayer-Uet/AI-local.git
cd AI-local

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# httpx
pip install httpx

# ollama setup
curl -fsSL https://ollama.com/install.sh | sh

# download qwen3:8b
ollama pull qwen3:8b
