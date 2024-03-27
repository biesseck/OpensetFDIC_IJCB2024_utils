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

### Compile RCNN
```
cd retinaface
make
```

### Download pre-trained model RetinaFace-R50

- Save file [retinaface-R50.zip](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing) to folder `retinaface/model`
```
cd retinaface/model
unzip retinaface/model/retinaface-R50.zip -d retinaface/model/retinaface-R50
```

### Run face detection script
```
python detect_crop_faces_retinaface_OpensetFDIC_IJCB2024.py --input_path /datasets2/3rd_OpensetFDIC_IJCB2024/validation_images
```
