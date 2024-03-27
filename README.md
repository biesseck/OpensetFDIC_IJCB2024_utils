# OpensetFDIC_IJCB2024_utils

### Main requirements
- Python 3.9
- CUDA 11.2

### Config environment
```
git clone https://github.com/biesseck/OpensetFDIC_IJCB2024_utils.git
cd OpensetFDIC_IJCB2024_utils

ENV=BOVIFOCR_OpensetFDIC_IJCB2024
conda env create -y --name $ENV --file environment.yml
conda activate $ENV
pip3 install -r requirements.txt
```

### Config RetinaFace
```
cd retinaface
make
```
