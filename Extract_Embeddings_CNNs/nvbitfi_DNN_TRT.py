import os
import h5py
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

DEBUG = 0
# disable jetson nano wathchdog: $ echo N > /sys/kernel/debug/gpu.0/timeouts_enabled
# enable jetson nano wathchdog: $ echo Y > /sys/kernel/debug/gpu.0/timeouts_enabled

class TRT_load_embeddings:
    def __init__(self, path_dir, layer_number, batch_size=1, layer_output_shape=(1,)) -> None:
        self.target_dtype = np.float32
        current_path = os.path.dirname(__file__)
        self.path_dir = os.path.join(current_path,path_dir)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.layer_results = []
        self.layer_number = layer_number
        self.onnx_model_name = "Layer_pytorch.onnx"
        self.TRT_model_name = "Layer_pytorch.rtr"
        self.TRT_output_shape = layer_output_shape

        # dataset_file = os.path.join(
        #     current_path,
        #     f"embeddings_batch_size_{self.batch_size}_layer_id_{self.layer_number}.h5",
        # )
        dataset_file = os.path.join(
            self.path_dir,
            f"inputs_layer.h5",
        )
        with h5py.File(dataset_file, "r") as hf:
            self.Input_dataset = np.array(hf["layer_input"])
            #self.Output_dataset = np.array(hf["layer_output"])
            # self.batch_size=np.array(hf['batch_size'])
            
        #self.layer_model = torch.load(model_file, map_location=torch.device("cpu"))
        with open(os.path.join(self.path_dir, self.TRT_model_name), "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.output = np.empty(self.TRT_output_shape, dtype = self.target_dtype) 
            if DEBUG: print(self.TRT_output_shape)
            if DEBUG: print(self.output.shape)
            batch = 0
            sample_image = self.Input_dataset[
                        batch * self.batch_size : batch * self.batch_size + self.batch_size
                    ]
            self.d_input = cuda.mem_alloc(1 * sample_image.nbytes)
            self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream = cuda.Stream()
            # warming up
            # output=self.__TRT_forward_function(sample_image)
            # end warming up

        if (DEBUG): print(len(self.Input_dataset))
        #print(len(self.Output_dataset))

        if (DEBUG): print((self.Input_dataset.shape))
        #print((self.Output_dataset.shape))
        # print(self.batch_size)

    def __TRT_forward_function(self,input):
        cuda.memcpy_htod_async(self.d_input, input, self.stream)
        # execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
        #out = self.output
        return (self.output)

    def TRT_layer_inference(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        #with torch.no_grad():
        for batch in range(0, int(np.ceil(max_batches))):
            img = self.Input_dataset[
                batch * self.batch_size : batch * self.batch_size + self.batch_size
            ]
            output=self.__TRT_forward_function(img)
            self.layer_results.append(output)

        embeddings_outputs = np.concatenate(self.layer_results)

        if DEBUG: print(embeddings_outputs.shape)

        log_path_file = os.path.join(
            self.path_dir,
            f"Output_layer.h5",
        )

        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "layer_output", data=embeddings_outputs, compression="gzip"
            )