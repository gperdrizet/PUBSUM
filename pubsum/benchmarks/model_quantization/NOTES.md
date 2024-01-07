# Notes

1. FlashAttention-2 not implemented by model
2. autoGPTQ does not support T5 models
3. AWQ only works on GPUs with compute >= 8.0
4. Attempting to load model with 8 bit quantization gives the following error:

```text
Error named symbol not found at line 529 in file /mmfs1/gscratch/zlab/timdettmers/git/bnb/csrc/ops.cu
```

Also relevant:

```text
$ python -m bnb

===================================BUG REPORT===================================
Welcome to bnb. For bug reports, please run

python -m bnb

 and submit this information together with your error trace to: https://github.com/TimDettmers/bnb/issues
================================================================================
bin /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/cuda_setup/main.py:149: UserWarning: /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/tensorrt_libs: did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('//ipv4.api.hosting.ionos.com/dns/v1/dyndns?q=YjAxMzNlMmMxZTI3NGVjN2JkNDQyOTlmMWYzMTU5ODQuUFE5N3Ywc0lORUJ5eGZDT3Z0WDlzSDlVN0pyVFFvX2VJdUthaWxGN0k3bjRVYmtTeFVLYTlKQUFHUW5KWUprQktDVWtDTm5sOXVGQVlKT09Gc0RJTGc'), PosixPath('https')}
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0'), PosixPath('/usr/local/cuda/lib64/libcudart.so')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 3.7
CUDA SETUP: Detected CUDA version 118
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/cuda_setup/main.py:149: UserWarning: WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!
  warn(msg)
CUDA SETUP: Loading binary /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libcuda.so

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda110.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda110_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda111.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda111_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda112.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda112_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda113.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda113_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda114.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda114_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda115.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda115_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda116.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda116_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda117.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda117_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda120.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda120_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda121.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda121_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libc10_cuda.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++
 /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/nvidia/cudnn/lib CUDA PATHS 

 /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/tensorrt_libs CUDA PATHS 


++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['3.7', '3.7', '3.7', '3.7']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable


WARNING: Please be sure to sanitize sensible info from any such env vars!

Error named symbol not found at line 117 in file /mmfs1/gscratch/zlab/timdettmers/git/bnb/csrc/ops.cu
```

First, looks like we are stuck with 'slow 8-bit matmul' because of the age of our GPUs, but we can test it anyway. Also 4-bit should work, in my reading on the bnb github repo, all GPUs support 4-bit.

Next, looks like bnb had some trouble finding libcudart.so seems like maybe it's because our CUDA install is not in the virtual environment. Seems like it gets found eventually, so leave it alone for now.

The thing with the ionos dynamic DNS directory in the path being non-existent is a real headscratcher. It's not in the PATH environment variable and has nothing to do with CUDA, bnb or Huggingface. Not sure how to even think about addressing that one. Again, hopefully not an actual problem. UPDATE: figured it out, was setting some environment variables via .bashrc that were no longer needed, removing them fixed the issue.

Some other issues present, but the last line brought me to the [this issue](https://github.com/TimDettmers/bnb/issues/566) on github. Following those instructions gives the following:

```text
$ python -m bnb

===================================BUG REPORT===================================
Welcome to bnb. For bug reports, please run

python -m bnb

 and submit this information together with your error trace to: https://github.com/TimDettmers/bnb/issues
================================================================================
bin /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 3.7
CUDA SETUP: Detected CUDA version 118
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/cuda_setup/main.py:149: UserWarning: WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!
  warn(msg)
CUDA SETUP: Loading binary /home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libcuda.so

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda110.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda110_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda111.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda111_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda112.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda112_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda113.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda113_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda114.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda114_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda115.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda115_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda116.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda116_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda117.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda117_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda118_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda120.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda120_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda121.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/bnb/libbnb_cuda121_nocublaslt.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/torch/lib/libc10_cuda.so
/home/siderealyear/arkk/huggingface_transformers/.env/lib/python3.8/site-packages/flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++
+++++++++++++ /usr/local/cuda/lib64 CUDA PATHS +++++++++++++


++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['3.7', '3.7', '3.7', '3.7']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable


WARNING: Please be sure to sanitize sensible info from any such env vars!

Error named symbol not found at line 117 in file /mmfs1/gscratch/zlab/timdettmers/git/bnb/csrc/ops.cu
```

Seems like bnb finds cuda more easily but problem persists - some others in the issue thread report the same.

After some further reading in the [github repo](https://github.com/TimDettmers/bitsandbytes), I noticed that the compile from source instructions mention using a different make target for kepler cards. See [here](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md). Following those instructions seems to fix the issue:

```text
$ python -m bnb

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libcuda.so

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++
/home/siderealyear/arkk/bnb/bnb/libbnb_cuda117.so
/home/siderealyear/arkk/bnb/bnb/libbnb_cuda118.so
/home/siderealyear/arkk/bnb/bnb/libbnb_cuda117_nocublaslt.so
/home/siderealyear/arkk/bnb/build/lib/bnb/libbnb_cuda117.so
/home/siderealyear/arkk/bnb/build/lib/bnb/libbnb_cuda118.so
/home/siderealyear/arkk/bnb/build/lib/bnb/libbnb_cuda117_nocublaslt.so

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++
+++++++++++++ /usr/local/cuda/lib64 CUDA PATHS +++++++++++++


++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['3.7', '3.7', '3.7', '3.7']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable


WARNING: Please be sure to sanitize sensible info from any such env vars!

SUCCESS!
Installation was successful!
```

Now the benchmark runs, but we have two more issues. During benchmark runs with no quantization we get the following warning:

```text
W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
```

Trivial fix - not even sure why or what is looking or TensorRT, as far as I know, we are not using tensorflow at all. Removing tensorflow and TensorRT from the environment silences the warning. No apparent issues.

Eight bit quantization works with no issues, but four bit complains:

```text
bnb/nn/modules.py:226: UserWarning: Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.
  warnings.warn(f'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_type=torch.float32 (default). This will lead to slow inference or training speed.')
```

Fixed by adding the following to model load:

```text
bnb_4bit_compute_dtype=torch.float16
```

Note: this is different from the torch.bfloat16 suggested in the [huggingface docs](https://huggingface.co/docs/transformers/main_classes/quantization).

## 5. Also, sometimes getting the following warning

```text
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Error named symbol not found at line 74 in file /mmfs1/gscratch/zlab/timdettmers/git/bitsandbytes/csrc/ops.cu
```

A little strange since we are not using multiprocessing for this benchmark - it should be a single job on a single GPU. Did sporadically see this warning when working on the parallel summarization benchmark too, never figured out what it was because it was not reliably reproducible.

Fixed by setting:

```text
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
```

From launcher script before run start.
