set -euo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl bash nodejs npm ripgrep ca-certificates
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
nvm install 22
nvm alias default 22
npm -v
npm install -g @openai/codex@0.154.0
codex --version
npm ls -g --depth=1 @openai/codex
