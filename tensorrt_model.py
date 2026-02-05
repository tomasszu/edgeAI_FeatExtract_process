import ctypes
import tensorrt as trt
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(None, "")

import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import os


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTModel:
    def __init__(self, engine_path, device_id=0):
        
        # šis ir pycuda autoinit aizvietotājs
        # Initialize CUDA driver (safe to call multiple times)
        cuda.init()
        self.device = cuda.Device(device_id)
        # create_context returns a pycuda.driver.Context and pushes it on the stack
        # Keep the reference so we can pop/detach it later
        self.cuda_ctx = self.device.make_context()


        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.input_memory = None
        self.output_memory = None
        self.input_memory_size = 0
        self.output_memory_size = 0
        self.output_buffer = None
        self.stream = cuda.Stream()

        # Initialize input/output tensor names
        
        self.input_tensor_name = 'input'  # Assuming the input tensor is named 'input'
        self.output_tensor_name = 'output'  # Assuming the output tensor is named 'output'

    def _load_engine(self, engine_path):
        assert os.path.exists(engine_path)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        batch_size = input_data.shape[0]
        self.context.set_input_shape(self.input_tensor_name, input_data.shape)

        # Allocate memory
        input_data = np.ascontiguousarray(input_data)
        input_nbytes = input_data.nbytes
        if self.input_memory is None or self.input_memory_size < input_nbytes:
            self.input_memory = cuda.mem_alloc(input_nbytes)
            self.input_memory_size = input_nbytes

        out_shape = self.context.get_tensor_shape(self.output_tensor_name)
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_tensor_name))
        out_size = trt.volume(out_shape)
        out_nbytes = out_size * np.dtype(out_dtype).itemsize
        if self.output_memory is None or self.output_memory_size < out_nbytes:
            self.output_buffer = cuda.pagelocked_empty(out_size, out_dtype)
            self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes)
            self.output_memory_size = out_nbytes

        self.context.set_tensor_address(self.input_tensor_name, int(self.input_memory))
        self.context.set_tensor_address(self.output_tensor_name, int(self.output_memory))

        # Run
        cuda.memcpy_htod_async(self.input_memory, input_data, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_buffer, self.output_memory, self.stream)
        self.stream.synchronize()

        output = np.array(self.output_buffer).reshape((batch_size, -1))
        norms = np.linalg.norm(output, axis=1, keepdims=True) + 1e-12
        return output / norms
    
    def destroy(self):
        # free device allocations
        try:
            if self.input_memory is not None:
                self.input_memory.free()
                self.input_memory = None
        except Exception:
            pass

        try:
            if self.output_memory is not None:
                self.output_memory.free()
                self.output_memory = None
        except Exception:
            pass

        # drop references to host buffer and TRT execution context/engine
        self.output_buffer = None
        try:
            # Try to explicitly delete the TRT execution context
            if getattr(self, "context", None) is not None:
                # in python bindings there is no pop/destroy method; delete ref
                del self.context
                self.context = None
        except Exception:
            pass

        try:
            if getattr(self, "engine", None) is not None:
                del self.engine
                self.engine = None
        except Exception:
            pass

        # Finally, pop and detach the CUDA context that we created
        try:
            if getattr(self, "cuda_ctx", None) is not None:
                # Pop the context off the stack and detach it
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None
        except Exception as e:
            # best-effort cleanup; log/ignore
            print("Error during CUDA context cleanup:", e)