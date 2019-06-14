# Settings for khz usage in one single gpu


## To run librispeech 
```bash
LINGVO_DIR="`pwd`/lingvo"  # (change to the cloned lingvo directory, e.g. "$HOME/lingvo")
LINGVO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
LIBRISPEECH_DIR="$HOME/librispeech" # saving librispeech part on this dir
sudo docker build --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04") - < ${LINGVO_DIR}/docker/dev.dockerfile
sudo docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ${LIBRISPEECH_DIR}:/tmp/librispeech -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash
```

## After entering into docker container

```shell
export CUDA_VISIBLE_DEVICES=0  
bazel test -c opt //lingvo:trainer_test //lingvo:models_test --test_verbose_timeout_warnings
```

To specify the GPU device helps the error concerns about:
* TBD : error for having multiple tasks running at the same time.
      : need to figure out where to modify the multiprocess configuration
      currently, just run multiple times so that each time one test could be executed successfully 

```bash
unable to create StreamExecutor for CUDA:0: failed initializing StreamExecutor for CUDA device ordinal 0: Internal: failed call to cuDevicePrimaryCtxRetain: CUDA_ERROR_INVALID_DEVICE: invalid device ordinal 0
```

## To fetch the training data & feature extraction

* Google uses 10 gpu for this feature extraction python script. And the bash file is hard coded for this configuration

- In my case, I have 1 gpu only. To configure the processor usage , modify the file : lingvo/tasks/asr/tools/librispeech.03.parameterize_train.sh 

```bash
for subshard in $(seq 0 0); do  # process id is 0
  set -x
  nice -n 20 $CMD \
    --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --generate_tfrecords \
    --transcripts_filepath="${ROOT}/train/${subset}.txt" \
    --shard_id="${subshard}" --num_shards=10 --num_output_shards=100 \
    --output_range_begin="${subshard}" --output_range_end="$((subshard + 10))" \   # and it compute file from 0-10
    --output_template="${ROOT}/train/train.tfrecords-%5.5d-of-%5.5d" || touch FAILED &
  set +x
done
```

```bash
lingvo/tasks/asr/tools/librispeech.01.download_train.sh  # bash file to fetch librispeech from official site saved as tar.gz
lingvo/tasks/asr/tools/librispeech.02.download_devtest.sh 
bazel build -c opt //lingvo/tools:create_asr_features   # building the extraction tool, this tool will be used in the following script
lingvo/tasks/asr/tools/librispeech.03.parameterize_train.sh # extracting the features , save only text and mfcc
```


## training cmd

```
bazel-bin/lingvo/trainer --enable_asserts=false --run_locally=cpu --mode=sync --model=asr.librispeech.Librispeech960Base --logdir=/tmp/librispeech/log --logtostderr
```
## debug reference
[use tensorboard](https://github.com/tensorflow/lingvo/issues/94)
```python
self._sess = tf_debug.TensorBoardDebugWrapperSession(self._sess, 'localhost:6008')
```
