# Setup notes

## PostgreSQL database server setup

### 1. Installation

First, get Ubuntu binary for version 16. We are on Ubuntu 20.04, which only provides version 12. To get the current version we need to add the postgresQL apt repository:

```
sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql
```

Install of 16 appears successful, no errors or warnings. The install process created a new user account, *postgres*, with home directory at */var/lib/postgresql*.

### 2. Database location

Next, we want to place our 'main' database on our RAID array - the solution to this is rather tricky to find because the Ubuntu documentation is non-existent as far as I can tell. It gives you a basic set-up and then refers you to the main postgreSQL docs - but the sys admin stuff for ubuntu is totally different, so none of the instructions work. Big pain in the butt - choices are: build it from source and configure everything manually (huge waste of time), use the Ubuntu binary and do the following to at least get the database off of the boot disk and to a place that makes sense.

The database directory can be specified from */etc/postgresql/16/main/postgresql.conf* which by default points to */var/lib/postgresql/16/main* with a running database (which is set-up for you during the apt install) that directory contains the following:

```
base          pg_dynshmem   pg_notify    pg_snapshots  pg_subtrans  PG_VERSION  postgresql.auto.conf
global        pg_logical    pg_replslot  pg_stat       pg_tblspc    pg_wal      postmaster.opts
pg_commit_ts  pg_multixact  pg_serial    pg_stat_tmp   pg_twophase  pg_xact     postmaster.pid
```

This means we cannot simply point postgresql at another directory - we need to copy this on first, then update the data directory in postgresql.conf. The copy location also needs a little bit of setup first too - the parent needs to be owned by the user *postgres* and have permission 0700. Let's do that now. From the grandparent of the intended new location of *main*:

```
user$ mkdir ./16
user$ sudo chown postgres:postgres ./16
user$ sudo chmod 0700 ./16
user$ sudo su postgres
postgres$ cp -r /var/lib/postgresql/16/main ./16/
postgres$ ls -l ./16

total 4
drwx------ 19 postgres postgres 4096 Sep 21 15:43 main

postgres$ exit

```

Now edit */etc/postgresql/16/main/postgresql.conf* and change *data_directory* to the full path of the new target *main* directory. Then restart the server and check status:

```
$ sudo systemctl restart postgresql.service
$ sudo systemctl status postgresql*

● postgresql@16-main.service - PostgreSQL Cluster 16-main
     Loaded: loaded (/lib/systemd/system/postgresql@.service; enabled-runtime; vendor preset: enabled)
     Active: active (running) since Thu 2023-09-21 15:47:25 UTC; 30s ago
    Process: 15066 ExecStart=/usr/bin/pg_ctlcluster --skip-systemctl-redirect 16-main start (code=exited, status=0/SUCCESS)
   Main PID: 15091 (postgres)
      Tasks: 6 (limit: 8792)
     Memory: 19.8M
     CGroup: /system.slice/system-postgresql.slice/postgresql@16-main.service
             ├─15091 /usr/lib/postgresql/16/bin/postgres -D /home/siderealyear/arkk/postgresql/16/main -c config_file=/etc/postgresql/>
             ├─15092 postgres: 16/main: checkpointer
             ├─15093 postgres: 16/main: background writer
             ├─15095 postgres: 16/main: walwriter
             ├─15096 postgres: 16/main: autovacuum launcher
             └─15097 postgres: 16/main: logical replication launcher

Sep 21 15:47:22 arkk systemd[1]: Starting PostgreSQL Cluster 16-main...
Sep 21 15:47:22 arkk postgresql@16-main[15066]: Removed stale pid file.
Sep 21 15:47:25 arkk systemd[1]: Started PostgreSQL Cluster 16-main.

● postgresql.service - PostgreSQL RDBMS
     Loaded: loaded (/lib/systemd/system/postgresql.service; enabled; vendor preset: enabled)
     Active: active (exited) since Thu 2023-09-21 15:47:25 UTC; 30s ago
    Process: 15108 ExecStart=/bin/true (code=exited, status=0/SUCCESS)
   Main PID: 15108 (code=exited, status=0/SUCCESS)

Sep 21 15:47:25 arkk systemd[1]: Starting PostgreSQL RDBMS...
Sep 21 15:47:25 arkk systemd[1]: Finished PostgreSQL RDBMS.
```

Looks good - note: there are two services, one is an umbrella service to start the actual database server and the other is the actual database server.

### 3. Network access

Last step in the basic generalized setup is to open up access to the database server for other machines on the local network. To do this we need to edit two configurations. First in */etc/postgresql/16/main/postgresql.conf* change the value of *listen_addresses* to the IP you want to listen on and change the value of *password_encryption* to *scram-sha-256* if desired.

Next, edit the file */etc/postgresql/16/main/pg_hba.conf* and add the following:

```
hostssl postgres       postgres        192.168.2.1/24        scram-sha-256
```

Then, set-up a password for the *postgres* database:

```
user$ sudo -u postgres psql postgres

psql (16.0 (Ubuntu 16.0-1.pgdg20.04+1))
Type "help" for help.
postgres=# ALTER USER postgres with encrypted password 'your_password';
ALTER ROLE

postgres=# exit
```

Lastly, make sure to let port 5432 through ufw on both the server and client machines and restart the database server

### 4. Test network access

To test network access, log into the client machine and try accessing the database via postgresql-client.

```
sudo apt install postgresql-client
$ psql --host 192.168.2.1  --username postgres --password --dbname postgres
Password: 
psql (12.16 (Ubuntu 12.16-0ubuntu0.20.04.1), server 16.0 (Ubuntu 16.0-1.pgdg20.04+1))
WARNING: psql major version 12, server major version 16.
         Some psql features might not work.
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
Type "help" for help.

postgres=#
```

Success!

## Set-up of HuggingFace Transformers with TensorFlow 2 GPU

**N.B.** NVIDIA driver 470 and CUDA 11.8 are already installed.

### 1. Create virtual environment

```
python -m venv .env
source .env/bin/activate
pip install --upgrade pip
```

### 2. Install TensorFlow and Transformers

```
pip install nvidia-cudnn-cu11==8.6.0.163
pip install tensorrt
pip install tensorflow==2.13.0
pip install transformers
```

### 3. Set environment variables

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

### 4. Test

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

### 5. Gotchas

#### a. Any time Tensorflow spins up we get the following output/warnings

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

#### b. This next one repeats ~10 times, the exact amount varies

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

#### c. Also, apparently from Transformers

```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
```

Note: instructions above were amended with the following fix. Explicitly supply default model and revision to pipeline:

```
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english', revision='af0f99b')('we love you'))"
```
