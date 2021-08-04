import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
import cv2
import imageio
import argparse

import tqdm
from icecream import ic

import os.path as osp
import sys
from pathlib import Path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.image_utils import save_img

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == "__main__":
    start_time_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--engine", nargs="?", type=str, default="./bin/glare_bestmodel_35_11_train_37_28.engine", help="engine file name",)
    parser.add_argument("--width", nargs="?", type=int, default=3584, help="target input width",)
    parser.add_argument("--height", nargs="?", type=int, default=2560, help="target input height", )
    parser.add_argument('--data_path', type=str, help='Training data path', default='/dataset_sub/camera_light_glare/patches_512_all/')
    parser.add_argument('--submission_dir', type=str, help='Training data path', default='./submission/tensorrt/')
    args = parser.parse_args()

    engine_NAME = args.engine
    proc_size = [args.width, args.height]
    total_time = 0; i = 0; s = 0

    ### Read the serialized ICudaEngine
    with open(engine_NAME, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        #print("Engine loaded!!!")

    context = engine.create_execution_context()
    buffers = allocate_buffers(engine, 1)
    
    H = 2448
    W = 3264
    total_elapsed_time = 0

    test_feature_paths = list(sorted(Path(args.data_path).glob("test_input_img/*.png")))

    ### Setting TensorRT buffer
    inputs, outputs, bindings, stream = buffers

    index = 0
    tq = tqdm.tqdm(enumerate(test_feature_paths), total=len(test_feature_paths))
    tq.set_description('Image # {}'.format(index))
    for index, (img_path) in tq:
        ### Pre-process ###
        start_time = time.perf_counter()
        new_fname = "test_" + img_path.name[11:-4]

        extended_in_feature = np.zeros((3,3584,2560))
        extended_in_feature[:,:W, :H] = (cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.).transpose(2, 1, 0)
        
        ### Input Image to TensorRT HOST buffer. 
        np.copyto(inputs[0].host, extended_in_feature.ravel())
        elapsed_time_pre_process = time.perf_counter() - start_time 

        ### Inference ###
        start_time = time.perf_counter()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        elapsed_time_inference = time.perf_counter() - start_time 
        
        ### Post-processing
        start_time = time.perf_counter()
        pred_img = (trt_outputs[0].reshape(3, proc_size[0], proc_size[1])[:,:W,:H]).transpose(2,1,0)*255.
        
        #save_img(osp.join(args.submission_dir + new_fname + '.png'),pred_img, color_domain='rgb')
        cv2.imwrite(osp.join(args.submission_dir + new_fname + '.png'),cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        elapsed_time_post = time.perf_counter() - start_time

        tq.set_postfix(time='Pre-Process={0:3.5f},Inference={1:3.5f}, Post-Process={2:3.5f}'.format(elapsed_time_pre_process, elapsed_time_inference, elapsed_time_post))
    tq.close() 

    total_elapsed_time_end = time.perf_counter() - start_time_start
    ic('Total elapsed time:', total_elapsed_time_end)