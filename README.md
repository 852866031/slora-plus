# S-LoRA

[S-LoRA](https://github.com/S-LoRA/S-LoRA)

If you have a fresh environment (ubuntu) do the following to start docker:

Install docker with [link](https://docs.docker.com/engine/install/ubuntu/)

Install NVIDIA driver with [link](https://ubuntu.com/server/docs/nvidia-drivers-installation)

Install NVIDIA container tool with [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Start docker:
```
sudo systemctl start docker
```
Then go to docker dir, run the following command to start docker 
```
sudo docker run -it --entrypoint /bin/bash --gpus all slora
```
If you want to start another shell inside docker, first check the name of the container with:
```
sudo docker ps
```
It will show the container names, copy the name and do:
```
sudo docker exec -it <container_name> bash
```

For test: 
```
python3 -m venv transformer_env
source transformer_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torchtext
pip install datasets
pip install tqdm
pip install numpy
pip install matplotlib
pip install pickle5
```