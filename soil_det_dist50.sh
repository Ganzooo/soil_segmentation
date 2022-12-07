batch_size=1
weight='/workspace/16_Demo/soil_segmentation/checkpoints/soil_segment_bestmodel_0_905_dist50.pth'
input_data_path='/dataset/Woodscape/soiling_dataset_nodist_50/' 
work_path='./cur/soil_segment_ddrnet23/soiling_dataset_nodist_50/'
nH=224
nW=320
save_img=True

python src/soil_detection.py -dp $input_data_path -bs $batch_size -t $weight --width $nW --height $nH --work_dir $work_path --save_image_val $save_img
python src/camera_failure_detection.py -dp $input_data_path -bs $batch_size -t $weight --width $nW --height $nH --work_dir $work_path --save_image_val $save_img