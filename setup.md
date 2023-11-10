# Set-up of HuggingFace Transformers with TensorFlow 2 GPU

**N.B.** NVIDIA driver 470 and CUDA 11.8 are already installed.

## 1. Create virtual environment

```
python -m venv .env
source .env/bin/activate
pip install --upgrade pip
```

## 2. Install TensorFlow and Transformers

```
pip install nvidia-cudnn-cu11==8.6.0.163
pip install tensorrt
pip install tensorflow==2.13.0
pip install transformers
```

## 3. Set environment variables

Add the following to *.env/bin/activate* so TensorFlow can find what it needs

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:`pwd`/.env/lib/python3.8/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
```

Also add the following to *.env/bin/activate* so Transformers knows where to put it's cache. In our case we create our cash via a symbolic link to a directory on an NVMe SSD used for fast scratch.

```
export TRANSFORMERS_CACHE=$(pwd)/.cache
```

```
mkdir .cache
ln -s /home/siderealyear/fast_scratch/huggingface_transformers_cache `pwd`/.cache
```

Deactivate and reactivate the virtual environment so the changes take effect.

## 4. Test

First test TensorFlow CPU and GPU:

```
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

tf.Tensor(1066.9298, shape=(), dtype=float32)
```

```
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
```

Then try out transformers:

```
$ python -c "from transformers import pipeline; print(pipeline('sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english')('we love you'))"

All PyTorch model weights were used when initializing TFDistilBertForSequenceClassification.

All the weights of TFDistilBertForSequenceClassification were initialized from the PyTorch model.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFDistilBertForSequenceClassification for predictions without further training.
[{'label': 'POSITIVE', 'score': 0.9998704195022583}]
```

Success! However, on first run we do have some warnings - see below for fixes.

## 5. Gotchas

### a. Any time Tensorflow spins up we get the following output/warnings

```
W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
```

Fixed with (note: instruction above amended with solution):

```
pip install tensorrt
```

Then adding the following to *.env/bin/activate*:

```
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:`pwd`/.env/lib/python3.8/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
```

Deactivate and reactivate virtual environment for changes to take effect.

### b. This next one repeats ~10 times, the exact amount varies

```
I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
```

This one is a bit more annoying - has to do with my system firmware not setting the numa_node files' values correctly at boot. leaving the fix here for future reference.

Find PCIe IDs of GPUs:

```
$ lspci -D | grep NVIDIA

0000:05:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
0000:06:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
0000:09:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
0000:0a:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
```

Set value in numa node file to 0 for each GPU at boot by adding the following sudo cron job:

```
@reboot (echo 0 | tee -a "/sys/bus/pci/devices/0000:05:00.0/numa_node")
@reboot (echo 0 | tee -a "/sys/bus/pci/devices/0000:06:00.0/numa_node")
@reboot (echo 0 | tee -a "/sys/bus/pci/devices/0000:09:00.0/numa_node")
@reboot (echo 0 | tee -a "/sys/bus/pci/devices/0000:0a:00.0/numa_node")
```

And reboot.

### c. Also, apparently from Transformers

```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
```

Note: instructions above were amended with the following fix. Explicitly supply default model and revision to pipeline:

```
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english', revision='af0f99b')('we love you'))"
```
