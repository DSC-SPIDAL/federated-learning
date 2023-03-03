# Flower Image Classification using PyTorch

This implementation of Flower uses PyTorch. Project does a distributed federated trainning.

## Project Setup

The project mainly consists of following files.  
```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
-- run.sh
```

Client and Server are respectively implemented in client.py and server.py. 
To execute the complete experiment `run.sh` can be used.
Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`
