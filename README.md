# Containerized CONQORD

## Building image

```
docker build -t conqord .
```

## Starting container

```
docker run -td --gpus=all --ipc=host conqord
```

## Running steps

```
export MODEL=Qwen/Qwen3-1.7B
export HF_TOKEN=<...>

sh conqord-container/scripts/run_steps.sh
```

## Running evaluation

```
export MODEL=Qwen/Qwen3-1.7B
export HF_TOKEN=<...>

sh conqord-container/scripts/run_evaluation.sh
```