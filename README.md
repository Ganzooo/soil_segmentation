# camera_lens_glare
camera_lens_glare_reduction

***Dataset setting***
1. Download dataset
WoodScape dataset: https://woodscape.valeo.com/download
[Download link](https://drive.google.com/uc?export=download&id=1Id-K7SjwCqWkLwtIGJUj5Q0Dw0E_TP_9)
2. Remove Layer Green
--> Add source file in repositery

***Train***
1. Install prerequested files
pip install -r requirements.txt

2. Train 
python src/train_no_apex.py --batch_size 4 --max_epoch 50 --data_path *[data_path]* --width 1280 --height 980 --model_type hardnet --work_dir ./cur/soil_detection/

ToDo:
[] Dataset modify 
[] Inference module add
