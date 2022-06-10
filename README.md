# UNet
UNet2D and UNet3D implement by PyTorch

---

### 自己按照需要，自行安装CPU或GPU版本的PyTorch

```shell
# GPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# CPU
pip3 install torch torchvision torchaudio
```

---

### 安装依赖包

```shell
pip install -r requirements.txt
```

### 项目结构

```shell
.
+---data
|   |  FruitFlyCell.py
|   |  Hippocampus.py
|   \  __init__.py
+---datasets
|   +---2d
|   |   \---cell
|   |       +---test
|   |       |   \---image
|   |       \---train
|   |           +---image
|   |           \---label
|   \---3d
|       \---hippocampus
|           +---test
|           |   \---imagesTs
|           \---train
|               +---imagesTr
|               \---labelsTr
+---model
|   |  unet2d.py
|   |  unet3d.py
|   \  __init__.py
+---utils
|   |  drawCurve.py
|   |  oneHot.py
|   |  test_all.png
|   \  __init__.py
|  README.md
|  requirements.txt
|  test2d.py
|  test3d.py
|  train2d.py
\  train3d.py
```
