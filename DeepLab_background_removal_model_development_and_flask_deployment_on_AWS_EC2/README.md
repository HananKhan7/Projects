# "DeepLabV3 Background Removal: Model Development and Flask Deployment on AWS EC2

## Introduction
This project demonstrates the use of DeepLabV3, a state-of-the-art deep learning model, for background removal in images. The model is trained to accurately segment the foreground object from the background, resulting in a transparent background. Additionally, the project showcases the deployment of the model using Flask on an AWS EC2 instance.

## Background Removal

Background removal is a common preprocessing step in image editing and computer vision tasks. DeepLabV3 is a convolutional neural network architecture designed for semantic image segmentation. In this project, we leverage the power of DeepLabV3 to accurately segment and remove backgrounds from images.

## Project Overview

- **Model Development**: Utilzied pre trained DeepLabV3 model on a suitable dataset for accurate background removal.
- **Flask Deployment**: Implemented a modular approach for model deployment using Flask allowing users to input image and get background-removed results.
- **AWS EC2 Deployment**: Deployed the Flask application on an AWS EC2 instance.

## Implementation
### Prerequisites
- Python prerequisites required for building and deploying the model are mentioned in "requirements.txt" file.
- An AWS accound is required for EC2 deployment.
- A couple of input images are provided in 'test_images' directory, but any other image can also be used.

### Running locally
- Run the Flask app locally:
'''python
python main.py
'''
- Postman API platform can be used for API calls.
- The final foreground image is stored in './final image/final_foreground_image.jpg'

### Deployment on AWS EC2
- The configurations used, apart from default, for setting up the EC2 instance are following:

- Amazon Machine Image (AMI) : Amazon Linux 2 AMI (HVM)

- Instance type: t2.small (Atleast)

- In network settings:
	- Allow HTTPS traffic from the internet (Allowed)
	- Allow HTTP traffic from the internet	(Allowed)
	- An additional security group rule was created of 'Custom TCP' type with a port range of 5000 (compatible with flask)

- Configure storage of 20 GiB was utilized (gp3 root volume)

#### Within EC2 instance

- EC2 instance was checked for updates
- git was installed to clone the repository (An SSH client can also be used) 
- Python virtual environment was created with libraries installed from 'requirements.txt' file.
- Deeplab model was deployed on EC2 instance, Postman API platform was used for API calls.

## Output

Final output contains a background-removed version of the input image.



