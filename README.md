```bash
conda create -y -n hearbridge-slr python=3.8
conda activate hearbridge-slr
conda install pytorch torchvision pytorch-cuda=12.1 cuda -c pytorch -c nvidia
conda install lintel -c conda-forge
pip install -r requirements.txt

pip install -U openmim
mim install mmengine

export CUDA_HOME=$CONDA_PREFIX
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```