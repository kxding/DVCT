# 配置环境
1. conda env create -f environment.yml
2. pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

3. 需要下载所需的model（如 stable diffusion v1.5)，并修改model_path

# 生成图片

在 gen.ipynb中生成, 在output中已经提供一些 训练好的embeddings