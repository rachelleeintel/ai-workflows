# **Vision-based Transfer Learning - Training**

## **Overview**
The vision-based Transfer Learning workflow showcases transfer learning on images to accomplish different classification tasks that range from binary classification to multiclass classification giving best performance on Intel Hardware utilizing the optimizations that could be done. For detailed information about the workflow, go to [End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) GitHub repository.

## **How it Works**

* The workflow uses pretrained SOTA models ( RESNET V1.5) from TF hub and Intel optimized TensorFlow to transfer the knowledge from a pretrained domain to a different custom domain achieving required accuracy

* Image classification is done on three domains: sports , medical imaging and remote sensing

* Uses AMX  BF16 in SPR which speeds up the training time significantly, without loss in accurac

* The workflow uses BF16 precision in SPR which speeds up the training time using Intel® AMX, without noticeable loss in accuracy when compared to FP32 precision using (Intel®  AVX-512)

![image](https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.e2e-vision-transfer-learning/assets/99835661/de8d7e76-50e4-42d0-8f83-72fdd96a0888)

## Get Started

### **Prerequisites**

#### Download the repo
Clone [End-to-End Vision Transfer Learning](https://github.com/intel/vision-based-transfer-learning-and-inference) repository into your working directory and switch to v1.0.0 Release branch.
```
git clone https://github.com/intel/vision-based-transfer-learning-and-inference.git .
git checkout v1.0.0
```

#### Download Dataset
##### Medical Imaging
Dataset is downloaded from TensorFlow website when the code is run for the first time. The dataset used for this domain is `colorectal_histology`. More details can be found at [Tensorflow Datases](https://www.tensorflow.org/datasets/catalog/colorectal_histology). 

##### Remote Sensing
The dataset used for this domain is [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45).  
[Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&action=locate ) the dataset and unzip the folder. The folder will later be used in `DATASET_DIR` when running the script.

### **Docker**
Below setup and how-to-run sessions are for users who want to use provided docker image.

#### Setup 

##### Quick Start Scripts
| Script Name | Description | 
| --- | --- |
| `colorectal.sh` | Script for medical imaging dataset | 
| `resisc.sh` | Script for remote sensing dataset | 

##### Customization Options
| Script Name | Description | 
| --- | --- |
| DATASET_DIR | Download the dataset the set it to `DATASET_DIR`. This directory is optional for Medical Imaging as this dataset will be downloaded from TensorFlow website when the code is run for the first time. Set to `DATASET_DIR=/data` for Medical Imaging | 
| PLATFORM | `SPR` and `None` are supported platforms | 
| PRECISION | `bf16` and `FP32 ` are supported precisions | 
| SCRIPT | `colorectal` and `resisc ` are available scripts names | 

##### Pull Docker Image
```
docker pull intel/ai-workflows:vision-transfer-learning-training
```
#### How to run
The snippet below shows a quick start script running with a single instance using the following options: `PLATFORM=None`, `PRECISION=FP32` and `SCRIPT=colorectal`.
```
export DATASET_DIR=/data
export OUTPUT_DIR=<directory where the output log files will be written>
export PLATFORM=None
export PRECISION=FP32
export SCRIPT=colorectal

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR}/${SCRIPT} \
  --env PLATFORM=${PLATFORM} \
  --env PRECISION=${PRECISION} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume /${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  --privileged --init -it \
  intel/ai-workflows:vision-transfer-learning-training \
  conda run --no-capture-output -n transfer_learning ./${SCRIPT}.sh
```

### **Bare Metal**
Below setup and how-to-run sessions are for users who want to use bare metal environment. 

#### Setup 
```
conda create -n transfer_learning python=3.8 --yes
conda activate transfer_learning
conda install -c conda-forge gperftools -y

Set conda path and LD_PRELOAD path

pip install -r requirements.txt

wget http://algo-buildstore.intel.com/builds/TF-DO/releases/spr-internal-releases/ww30-gold-release/tf_nightly-2.10.0-cp38-cp38-linux_x86_64.whl
pip install tf_nightly-2.10.0-cp38-cp38-linux_x86_64.whl
```


#### How to run

   ##### 1) Sports Dataset Training
        a) FP32 : bash sports.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/sportsFP32/" --DATASET_DIR datasets/sports --PLATFORM SPR
        b) BF16: bash sports.sh --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/sportsBF16/" --DATASET_DIR datasets/sports --PLATFORM SPR
   ##### 2) Sports Dataset Inference
        a) Inference FP32: bash sports.sh --inference -cp "logs/fit/sportsFP32" --PRECISION FP32 --OUTPUT_DIR "logs/fit/trailinfFP32" --DATASET_DIR datasets/sports --PLATFORM SPR
        b) Inference BF16: bash sports.sh --inference -cp "logs/fit/sportsBF16" --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/trailinfBF16/" --DATASET_DIR datasets/sports --PLATFORM SPR

   ##### 2) Remote Sensing Dataset Training
        a) FP32 : bash resisc.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/resiscFP32/" --DATASET_DIR datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256
        b) BF16: bash resisc.sh --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/resiscBF16/" --DATASET_DIR  datasets/resisc45 --PLATFORM SPR --BATCH_SIZE 256
   
   ##### 2) Remote Sensing Dataset Inference
        a) Inference FP32: bash resisc.sh --inference -cp "logs/fit/resiscFP32" --PLATFORM SPR --DATASET_DIR datasets/resisc45
        b) Inference BF16: bash resisc.sh --PRECISION Mixed_Precision --inference -cp "logs/fit/resiscBF16" --PLATFORM SPR --DATASET_DIR datasets/resisc45


   ##### 3) Medical Imaging Dataset Training
        a) FP32 : bash colorectal.sh --PRECISION FP32 --OUTPUT_DIR "logs/fit/colorectalFP32/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
        b) BF16: bash colorectal.sh --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/colorectalBF16/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
   
   ##### 3) Medical Imaging Dataset Inference
        a) Inference FP32: bash colorectal.sh --inference -cp "logs/fit/colorectalFP32" --PRECISION FP32 --OUTPUT_DIR "logs/fit/colorectalFP32/" --DATASET_DIR datasets/colorectal --PLATFORM SPR
        b) Inference BF16: bash colorectal.sh --inference -cp "logs/fit/colorectalBF16" --PRECISION Mixed_Precision --OUTPUT_DIR "logs/fit/colorectalBF16/" --DATASET_DIR datasets/colorectal --PLATFORM SPR

## Documentation
[Docker* Repository](https://hub.docker.com/u/intel) <br>
[Main GitHub*](https://github.com/intel/vision-based-transfer-learning-and-inference)<br>
[Readme](https://github.com/intel/vision-based-transfer-learning-and-inference/blob/main/README.md)<br>
[Release Notes](https://github.com/intel/vision-based-transfer-learning-and-inference/releases/tag/v1.0.0)<br>

## Code Sources
[Dockerfile](https://github.com/intel/ai-workflows/blob/v0.1.0/transfer_learning/tensorflow/resnet50/training/Dockerfile.vision-transfer-learning)<br>
[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)<br>

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license file](https://github.com/intel/ai-workflows/blob/main/LICENSE) for additional details.

## Related Containers and Solutions
[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
