# camera_lens_glare
camera_lens_glare_reduction

-> log files saved in work_dir
-> saved image files saved in reuslt_dir
-> submission files saved in submission_dir

1. Install warmup scheduler
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

2. Install prerequested files
pip install -r requirements.txt

3. Patch generation:
python src/dataset/generate_patches.py --src_dir/dataset_sub/camera_light_glare/train/ --tar_dir /dataset_sub/camera_light_glare/patches_1024_sample/train_patch/ --ps 256  --num_patches 20

4. Train 
CUDA_VISIBLE_DEVICES=0 python src/train_no_apex.py --batch_size 4 --max_epoch 50 --data_path /home/kt05/work/dataset/camera_light_glare/ --width 256 --height 256 --model_type resnet_unet --result_dir ./result_mirnet_no_pre/ --work_dir ./cur/deglare_mirnet_no_pre/ --submission_dir ./submission_mirnet/ 
