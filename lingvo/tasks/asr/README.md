# Settings for khz usage in one single gpu


## To run librispeech 
```
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
```
unable to create StreamExecutor for CUDA:0: failed initializing StreamExecutor for CUDA device ordinal 0: Internal: failed call to cuDevicePrimaryCtxRetain: CUDA_ERROR_INVALID_DEVICE: invalid device ordinal 0
```

## To fetch the training data + feature extraction

```
lingvo/tasks/asr/tools/librispeech.01.download_train.sh  # bash file to fetch librispeech from official site saved as tar.gz
lingvo/tasks/asr/tools/librispeech.02.download_devtest.sh 
bazel build -c opt //lingvo/tools:create_asr_features   # building the extraction tool, this tool will be used in the following script
lingvo/tasks/asr/tools/librispeech.03.parameterize_train.sh # extracting the features , save only text and mfcc



```
