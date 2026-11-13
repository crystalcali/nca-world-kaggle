# NCA World — Kaggle runner (initial connectivity test)

This repository contains the **restricted** Kaggle-side worker for our JEPA + Neural Cellular Automata research. The initial job is only a PyTorch CUDA smoke test. **No training has been started yet.**

## Setup in Kaggle

1. Create a Kaggle Notebook; in **Settings** select **GPU T4 x2** (or another GPU) and enable **Internet**.
2. In GitHub, create a **fine-grained personal access token** restricted to this repository only:
   - Repository contents: **Read-only**
   - Issues: **Read and write**
   - Metadata: Read (automatic)
   Do not use a classic full-access token.
3. In Kaggle, open **Add-ons → Secrets** (the interface may vary); create a secret named `GITHUB_TOKEN`, paste the token there and enable notebook access. **Do not put the token in a cell, code, or chat.**
4. Paste the cell below and run it:

```python
from kaggle_secrets import UserSecretsClient
import base64
import os
import requests

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo = "crystalcali/nca-world-kaggle"
url = f"https://api.github.com/repos/{repo}/contents/agent.py"
response = requests.get(
    url,
    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    timeout=30,
)
response.raise_for_status()
agent_source = base64.b64decode(response.json()["content"]).decode("utf-8")
os.environ["GITHUB_TOKEN"] = token
exec(compile(agent_source, "agent.py", "exec"), {"__name__": "__main__"})
```

The agent polls GitHub for up to 10 minutes, executes **one** approved job, publishes results as a comment on that job's GitHub issue, and exits. Kaggle may reclaim idle sessions; start the cell again for new jobs. The agent does not run arbitrary commands received from GitHub Issues.

## Safety & scope

- The agent only accepts issues created by `crystalcali` with a title starting `[COLAB-JOB]`, body exactly `{"task": "gpu_smoke"}`, and without a prior claim/result comment.
- It does not run shell scripts or arbitrary issue commands. Extend the allowlist only after reviewing the implementation.
- A token scoped to this one repository limits damage if credentials are compromised. Revoke it when no longer needed.
- **Currently this repository is public.** Change it to private if you do not want source code or results to be visible to others.
- Colab/Kaggle GPU sessions are time-limited. This is not an always-on autonomous service.
