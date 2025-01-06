# Open-Vocabulary Detection and Segmentation

![detection and segmentation demo](output.gif)

## Introduction

This example shows how to use the OWL-Vit and SAM to perform object detection and segmentation on an image/video using input text prompts. 

## Setup 
1. Clone this repo 
```
git clone https://github.com/affanrasheed/GenerativeAI_on_Jetson.git
cd GenerativeAI_on_Jetson/Open_Vocabulary_Segmentation
```
2. Install dependencies

	a. Install PyTorch
	
	b. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
	```
	git clone https://github.com/NVIDIA-AI-IOT/torch2trt
	cd torch2trt
	python setup.py install --user
	cd ..
	```
	
	c. Install NVIDIA TensorRT
	
	d. Install the Transformers library
	```
	python3 -m pip install transformers
	```
	
	e. Install the NanoOWL and NanoSAM package
	```
	python3 setup.py develop --user
	```
	
	f. Install tensorflow
	```
	pip3 install tensorflow
	```

3. Build the TensorRT engine for the OWL-ViT vision encoder
```
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine \
        data/owl_image_encoder_patch32.engine
```

4. Build the TensorRT engine for the Mask Decoder
Copy the MobileSAM mask decoder onnx file into data directory. Download onnx file from [here](https://drive.google.com/drive/folders/1YI7y5eM2n4HQ49moo-D1Db3JgincT0L1?usp=sharing)
```
trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10
```
5. Build the TensorRT engine for the NanoSAM encoder
Copy the NanoSAM mask encoder onnx file into data directory. Download onnx file from [here](https://drive.google.com/drive/folders/1YI7y5eM2n4HQ49moo-D1Db3JgincT0L1?usp=sharing)
```
trtexec \
    --onnx=data/resnet18_image_encoder.onnx \
    --saveEngine=data/resnet18_image_encoder.engine \
    --fp16
```

## OpenVocabulary Segmentation Demo

To launch the openvocabulary demo on its own, you can run the tree_demo.py script. Make sure camera is connected with the device: 

```
python3 tree_demo.py 
```

Once the script is launched, the UI will become available at ```http://localhost:7860```



