import os,subprocess,time,json
stages=[('apt', 'apt-get update && apt-get install -y curl bash nodejs npm ripgrep'), ('toolchain', 'set -euo pipefail; if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then  npm install -g @openai/codex@0.154.0; else  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&  export NVM_DIR="$HOME/.nvm" &&  \\. "$NVM_DIR/nvm.sh" || true &&  command -v nvm &>/dev/null || { echo \'Error: NVM failed to load\' >&2; exit 1; } &&  nvm install 22 && nvm alias default 22 && npm -v &&  npm install -g @openai/codex@0.154.0; fi && codex --version')]
for name,command in stages:
    start=time.monotonic()
    print('PREFLIGHT_STAGE_START',name,flush=True)
    result=subprocess.run(['bash','-lc',command],env={**os.environ,'NVM_NODEJS_ORG_MIRROR':'https://nodejs.org/dist'})
    print('PREFLIGHT_STAGE_END',json.dumps({'stage':name,'seconds':time.monotonic()-start,'returncode':result.returncode}),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
