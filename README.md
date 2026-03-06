# DeltaServe

Extend on [S-LoRA](https://github.com/S-LoRA/S-LoRA)

```
conda create -n dserve python=3.9
conda activate dserve 
# cuda > 12.6
pip install torch==2.8.0
pip install uvloop=0.22.0
pip install -e . --no-build-isolation
pip install triton==3.4.0
```

## Llama3 Experiments:
Please update the path in the following files to correct path in your system
```
S-LoRA/test/llama3/launch_llama3.py
S-LoRA/test/llama3/config/finetuning_config.json
S-LoRA/test/llama3/config/no_finetuning_config.json
```

Use the following command to generate two dummy LoRA adapters for experiments
```
cd S-LoRA/test/llama3
python adapter_train.py
cp -r adapters/llama3-toy-lora adapters/llama3-toy-lora-ft
```

Launch a server with llama3 model loaded on singe GPU:
```
cd S-LoRA/test/llama3
python launch_llama3.py
```

Auto benchmark that uses a timeline file to generate requests and feed to the server
```
cd S-LoRA/test/llama3
# For co-serving
python auto_benchmark.py --co
# For inference only
python auto_benchmark.py
```
