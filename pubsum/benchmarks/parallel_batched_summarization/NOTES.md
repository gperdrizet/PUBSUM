# Notes

## 1. 

After completing 3 jobs per GPU with batch size of 4 (resulting OOM for every replicate), received the following error on the first 3 jobs per GPU with batch size 8 job:

```
Traceback (most recent call last):
  File "run_benchmarks.py", line 261, in <module>
    parallel_batched.benchmark(
  File "/home/siderealyear/arkk/huggingface_transformers/pubsum/benchmarks/parallel_batched_summarization/benchmark.py", line 143, in benchmark
    result = [async_result.get() for async_result in async_results]
  File "/home/siderealyear/arkk/huggingface_transformers/pubsum/benchmarks/parallel_batched_summarization/benchmark.py", line 143, in <listcomp>
    result = [async_result.get() for async_result in async_results]
  File "/usr/lib/python3.8/multiprocessing/pool.py", line 771, in get
    raise self._value
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
```

## 2.

Running 3 jobs per GPU with four bit nf4 quantization is causing some kind of GPU crash. Some jobs will complete, seems like it might be when we start to loop on replicates. Watched NVIDIA-SMI during one of the crashes. Temps were file, all below 60 degrees C, memory use per chip was around 4 GB out of 11 GB and power draw was only ~ 100 W per chip. Not sure what could be causing in - NVIDIA-SMI suddenly outputs:

```text
Unable to determine the device handle for GPU 0000:05:00.0: Unknown Error
```

No errors or warnings or any thing at all coming from python, just hangs like it's still working but never finishes. System is not locked, but one zombie core is stuck at 100% running the benchmark script. Reboot clears it up, but takes an unusually long time, > 5 min.

Not sure what to do in the absence of a stack trace - would have to dig into system logs. Maybe let's not use nf4 and see if we get the same issue.

Still happening without nf4 quantization - was able to complete 2 replicates, on the third cuda:0 and cuda:1 never finished any workunits and nvidia-smi spit out the same error. Here is some more information:

```bash
$ nvidia-debugdump --list
Found 4 NVIDIA devices
Error: nvmlDeviceGetHandleByIndex(): Unknown Error
FAILED to get details on GPU (0x0): Unknown Error
```

```bash
$ sudo nvidia-bug-report.sh

nvidia-bug-report.sh will now collect information about your
system and create the file 'nvidia-bug-report.log.gz' in the current
directory.  It may take several seconds to run.  In some
cases, it may hang trying to capture data generated dynamically
by the Linux kernel and/or the NVIDIA kernel module.  While
the bug report log file will be incomplete if this happens, it
may still contain enough data to diagnose your problem.

If nvidia-bug-report.sh hangs, consider running with the --safe-mode
and --extra-system-data command line arguments.

Please include the 'nvidia-bug-report.log.gz' log file when reporting
your bug via the NVIDIA Linux forum (see forums.developer.nvidia.com)
or by sending email to 'linux-bugs@nvidia.com'.

By delivering 'nvidia-bug-report.log.gz' to NVIDIA, you acknowledge
and agree that personal information may inadvertently be included in
the output.  Notwithstanding the foregoing, NVIDIA will use the
output only for the purpose of investigating your reported issue.

Running nvidia-bug-report.sh... complete.
```

Here is a relevant snipit from the log:

```bash
Jan 16 14:04:39 pyrite kernel: [44643.002792] NVRM: GPU at PCI:0000:05:00: GPU-97363126-e553-8dd4-9ad8-56844b42694a
Jan 16 14:04:39 pyrite kernel: [44643.002803] NVRM: GPU Board Serial Number: 0325015054694
Jan 16 14:04:39 pyrite kernel: [44643.002804] NVRM: Xid (PCI:0000:05:00): 79, pid=1173, GPU has fallen off the bus.
Jan 16 14:04:39 pyrite kernel: [44643.002806] NVRM: GPU 0000:05:00.0: GPU has fallen off the bus.
Jan 16 14:04:39 pyrite kernel: [44643.002828] NVRM: GPU 0000:05:00.0: GPU serial number is 0325015054694.
Jan 16 14:04:39 pyrite kernel: [44643.002840] NVRM: A GPU crash dump has been created. If possible, please run
Jan 16 14:04:39 pyrite kernel: [44643.002840] NVRM: nvidia-bug-report.sh as root to collect this data before
Jan 16 14:04:39 pyrite kernel: [44643.002840] NVRM: the NVIDIA kernel module is unloaded.
Jan 16 14:04:39 pyrite kernel: [44643.002854] NVRM: GPU at PCI:0000:06:00: GPU-86d7b114-cfa7-c7bb-97c7-e76781232889
Jan 16 14:04:39 pyrite kernel: [44643.002865] NVRM: GPU Board Serial Number: 0325015054694
Jan 16 14:04:39 pyrite kernel: [44643.002866] NVRM: Xid (PCI:0000:06:00): 79, pid=1173, GPU has fallen off the bus.
Jan 16 14:04:39 pyrite kernel: [44643.002867] NVRM: GPU 0000:06:00.0: GPU has fallen off the bus.
Jan 16 14:04:39 pyrite kernel: [44643.002885] NVRM: GPU 0000:06:00.0: GPU serial number is 0325015054694.
```

Nvidia (developer forums suggest)[https://forums.developer.nvidia.com/t/unable-to-determine-the-device-handle-for-gpu-xxxxxxxx-unknown-error/230277] that Xid (fallen off the bus) could be caused by peak power issues - however, I know that our PSU can handel these cards. Or at least, it could. One thing I did notice is that the cards seem to stay in power state P0 during the run, not P8 like I'm pretty sure they did previously. Ok, interesting - turns out power states P0 and P8 are not what I thought they were. From the (nvidia documentation)[https://docs.nvidia.com/gameworks/content/gameworkslibrary/coresdk/nvapi/group__gpupstate.html]:

```text
The GPU performance state APIs are used to get and set various performance levels on a per-GPU basis. P-States are GPU active/executing performance capability and power consumption states.

P-States range from P0 to P15, with P0 being the highest performance/power state, and P15 being the lowest performance/power state. Each P-State maps to a performance level. Not all P-States are available on a given system. The definition of each P-States are currently as follows:

P0/P1 - Maximum 3D performance
P2/P3 - Balanced 3D performance-power
P8 - Basic HD video playback
P10 - DVD playback
P12 - Minimum idle power consumption
```

So if we are consistently in power state P0 - that's probably good for performance, but we may actually be drawing more peak power than we did perviously when we ran in P8 all the time. Two things to do:

1. Reset watt meter peak and let a run crash - see what our peak draw is.
2. Try to lock P2/P3 or P8 and see what our peak draw is and if we crash.

Also, let's keep an eye on which card is falling off the bus. I wonder if it's the one with the damaged connecter?

OK, two observations:

1. During a crashed run, the highest power draw recorded was ~630 W for both systems together. The K80s are running off of a 1000 W power supply, so unless something is broken or there was a momentary peak much higher that the watt meter missed, I can't imagine the GPU is dropping because of power concerns.

2. The GPUs appear to start the run in power state P8 and then flip to P0 once things ramp up.

Next let's try locking the clock rate down as suggested in the forum. Our K80s have a 560 MHz base clock and boost from 562 to 875 MHz, so let's try locking them at the base clock:

```bash
$ sudo nvidia-smi --lock-gpu-clocks=560

Setting locked GPU clocks is not supported for GPU 00000000:05:00.0.
Treating as warning and moving on.
Setting locked GPU clocks is not supported for GPU 00000000:06:00.0.
Treating as warning and moving on.
Setting locked GPU clocks is not supported for GPU 00000000:09:00.0.
Treating as warning and moving on.
Setting locked GPU clocks is not supported for GPU 00000000:0A:00.0.
Treating as warning and moving on.
All done.
```

OK, whelp, let's try it this way? Query supported clocks:

```bash
$ nvidia-smi -q -d SUPPORTED_CLOCKS

==============NVSMI LOG==============

Timestamp                                 : Wed Jan 17 00:27:14 2024
Driver Version                            : 470.42.01
CUDA Version                              : 11.4

Attached GPUs                             : 4
GPU 00000000:05:00.0
    Supported Clocks
        Memory                            : 2505 MHz
            Graphics                      : 875 MHz
            Graphics                      : 862 MHz
            Graphics                      : 849 MHz
            Graphics                      : 836 MHz
            Graphics                      : 823 MHz
            Graphics                      : 810 MHz
            Graphics                      : 797 MHz
            Graphics                      : 784 MHz
            Graphics                      : 771 MHz
            Graphics                      : 758 MHz
            Graphics                      : 745 MHz
            Graphics                      : 732 MHz
            Graphics                      : 719 MHz
            Graphics                      : 705 MHz
            Graphics                      : 692 MHz
            Graphics                      : 679 MHz
            Graphics                      : 666 MHz
            Graphics                      : 653 MHz
            Graphics                      : 640 MHz
            Graphics                      : 627 MHz
            Graphics                      : 614 MHz
            Graphics                      : 601 MHz
            Graphics                      : 588 MHz
            Graphics                      : 575 MHz
            Graphics                      : 562 MHz
        Memory                            : 324 MHz
            Graphics                      : 324 MHz
```

OK, that's probably why our clock lock did not work - the 'base' clock rate of 560 MHz doesn't show up as supported. I wonder why? Let's try setting the memory to 2505 MHz and the graphics to 562 MHz:

```bash
$ sudo nvidia-smi -ac 2505,562

Applications clocks set to "(MEM 2505, SM 562)" for GPU 00000000:05:00.0
Applications clocks set to "(MEM 2505, SM 562)" for GPU 00000000:06:00.0
Applications clocks set to "(MEM 2505, SM 562)" for GPU 00000000:09:00.0
Applications clocks set to "(MEM 2505, SM 562)" for GPU 00000000:0A:00.0
All done.
```
Ok, let's try it - also start logging in a screen:

```bash
$ screen -S nvidia-smi_dmon
$ nvidia-smi dmon --select pucet --filename four_bit_nf4_crash.log
```

OK, right off the bat, seems like our clock lock didn't hold. But, we got an error message this time. Here are the last few lines of output from the run_benchmarks.py script:

```text
Parallel batched summarization:

 Replicate: 3
 Model quantization: four bit
 Batch size: 1
 Batches: 3
 Workers per GPU: 1

 Job 0: starting on cuda:0.
 Job 1: starting on cuda:1.
 Job 2: starting on cuda:2.
 Job 3: starting on cuda:3.
 Job 0: summarizing batch 1 of 3.
 Job 1: summarizing batch 1 of 3.
 Job 2: summarizing batch 1 of 3.
 Job 3: summarizing batch 1 of 3.
 Job 2: finished batch 1 of 3.
 Job 2: summarizing batch 2 of 3.
 Job 0: finished batch 1 of 3
 Job 0: summarizing batch 2 of 3.
CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

 Job 3: finished batch 1 of 3
 Job 3: summarizing batch 2 of 3.
 Job 2: finished batch 2 of 3
 Job 2: summarizing batch 3 of 3.
 Job 3: finished batch 2 of 3
 Job 3: summarizing batch 3 of 3.
 Job 2: finished batch 3 of 3
 Job 2: done.
 Job 3: finished batch 3 of 3
 Job 3: done.
^CProcess ForkPoolWorker-1:19:
Process ForkPoolWorker-1:20:
Traceback (most recent call last):
  File "./run_benchmarks.py", line 249, in <module>
    p.join()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 149, in join
    res = self._popen.wait(timeout)
  File "/usr/lib/python3.8/multiprocessing/popen_fork.py", line 47, in wait
    return self.poll(os.WNOHANG if timeout == 0.0 else 0)
  File "/usr/lib/python3.8/multiprocessing/popen_fork.py", line 27, in poll
    pid, sts = os.waitpid(self.pid, flag)
KeyboardInterrupt
^CError in atexit._run_exitfuncs:
Traceback (most recent call last):
Process Process-1:
  File "/usr/lib/python3.8/multiprocessing/popen_fork.py", line 27, in poll
pid, sts = os.waitpid(self.pid, flag)
KeyboardInterrupt
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/mnt/arkk/huggingface_transformers/pubsum/benchmarks/parallel_batched_summarization/benchmark.py", line 182, in benchmark
    pool.join()
  File "/usr/lib/python3.8/multiprocessing/pool.py", line 662, in join 
    self._worker_handler.join()
  File "/usr/lib/python3.8/threading.py", line 1011, in join
    self._wait_for_tstate_lock()
  File "/usr/lib/python3.8/threading.py", line 1027, in _wait_for_tstate_lock
    elif lock.acquire(block, timeout):
KeyboardInterrupt

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 318, in _bootstrap
    util._exit_function()
  File "/usr/lib/python3.8/multiprocessing/util.py", line 334, in _exit_function
    _run_finalizers(0)
  File "/usr/lib/python3.8/multiprocessing/util.py", line 300, in _run_finalizers
    finalizer()
  File "/usr/lib/python3.8/multiprocessing/util.py", line 224, in __call__
    res = self._callback(*self._args, **self._kwargs)
  File "/usr/lib/python3.8/multiprocessing/pool.py", line 729, in _terminate_pool
    p.join()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 149, in join
    res = self._popen.wait(timeout)
  File "/usr/lib/python3.8/multiprocessing/popen_fork.py", line 47, in wait
    return self.poll(os.WNOHANG if timeout == 0.0 else 0)
  File "/usr/lib/python3.8/multiprocessing/popen_fork.py", line 27, in poll
    pid, sts = os.waitpid(self.pid, flag)
KeyboardInterrupt
```
OK, not helpful - we have a CUDA unknown error and nothing interesting in the logs. Here are the last few samples from before the crash:

```text
# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk sbecc dbecc   pci rxpci txpci
# Idx     W     C     C     %     %     %     %   MHz   MHz  errs  errs  errs  MB/s  MB/s
    0    73    37     -    29     4     0     0  2505   875     0     0     0     -     -
    1    85    46     -    29     4     0     0  2505   849     0     0     0     -     -
    2    71    36     -    29     4     0     0  2505   849     0     0     0     -     -
    3    88    47     -    29     4     0     0  2505   849     0     0     0     -     -
    0    73    37     -    28     4     0     0  2505   875     0     0     0     -     -
    1    84    46     -    29     4     0     0  2505   849     0     0     0     -     -
    2   128    37     -    98    27     0     0  2505   875     0     0     0     -     -
    3    87    48     -    29     4     0     0  2505   849     0     0     0     -     -
    0    73    38     -    29     4     0     0  2505   875     0     0     0     -     -
    1    84    46     -    29     4     0     0  2505   849     0     0     0     -     -
    2    74    36     -    28     4     0     0  2505   875     0     0     0     -     -
    3    88    48     -    30     4     0     0  2505   849     0     0     0     -     -
    0     -     -     -
```

Kinda stumped here. Seems like it's probably a CUDA think rather than a power/hardware issue. Just for fun, let's log a good run, not using quantization.

Yep - the CUDA + quantization hypothesis just went out the window. Unquantized job crashed almost immediately in exactly the same way. CUDA error: unknown error but everything else looks fine. Here are the last few sample from the nvidia-smi dmon log:

```text
# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk sbecc dbecc   pci rxpci txpci
# Idx     W     C     C     %     %     %     %   MHz   MHz  errs  errs  errs  MB/s  MB/s
    0    93    46     -    99    42     0     0  2505   653     0     0     0     -     -
    1   140    54     -    97    52     0     0  2505   875     0     0     0     -     -
    2   129    44     -    97    49     0     0  2505   875     0     0     0     -     -
    3   148    54     -    97    49     0     0  2505   875     0     0     0     -     -
    0    96    46     -    99    44     0     0  2505   692     0     0     0     -     -
    1   139    54     -    97    51     0     0  2505   875     0     0     0     -     -
    2   129    45     -    97    49     0     0  2505   875     0     0     0     -     -
    3   148    55     -    97    50     0     0  2505   875     0     0     0     -     -
    0    97    46     -    97    43     0     0  2505   692     0     0     0     -     -
    1   140    54     -    97    52     0     0  2505   875     0     0     0     -     -
    2   129    45     -    99    51     0     0  2505   875     0     0     0     -     -
    3   146    55     -    97    49     0     0  2505   862     0     0     0     -     -
    0     -     -     -
```

So, we were hitting the cards a bit harder, basically touching the power limit in some cases - but temps were fine and no memory errors. Starting to feel like at least one of these cards is cooked. Let's try a full poweroff and reseat the cards, then a clean boot and a run with no quantization. Failing that, we might need to (hopefully) determine which card is causing the problem and just pull it and run on one K80.