# Lay Summary Optimization with RLHF

Lay summary optimization with biomistral using PPO on metrics

**Task URL:** https://biolaysumm.org/

**Tech Stack:** pdm, pytorch lightning, unsloth, huggingface, mistral, trl, wandb, github actions

### Setup/Install

Run `make`

### Train

Run `pdm start`

### Running python scripts

Run `pdm run <script path>`, e.g. `pdm run src/burrito/main.py` or `pdm run jupyter lab`

### Adding dependencies

Run `pdm add <package name>`, e.g. `pdm add torch` or `pdm add "git+https://github.com/Dao-AILab/flash-attention"`

### Running linters/formatters

Linter: `pdm run lint`

Formatters: `pdm run lint-format`
