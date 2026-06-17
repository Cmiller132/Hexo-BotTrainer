"""Fetch the BC bootstrap corpus (spec §7 Phase A) to data/hexo-bootstrap-corpus.

Run with the hexfield-dev venv (huggingface_hub lives there, never in the
live build venv): /root/.venvs/hexfield-dev/bin/python scripts/_hexfield_fetch_corpus.py
"""

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="timmyburn/hexo-bootstrap-corpus",
    repo_type="dataset",
    local_dir="/mnt/e/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus",
)
print("downloaded to", path)
