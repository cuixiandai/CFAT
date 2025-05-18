# CFAT: Convolutional Fieldy Attention Transformer for Polarimetric SAR Image Classification

## Abstract

Classification of PolSAR (Polarimetric Synthetic Aperture Radar) images is one of the most prominent topics in the field of PolSAR and is crucial for its applications. In recent years, with the widespread application of deep learning technologies, new methods based on neural networks have continually emerged. However, due to the inherent complexity of PolSAR images, such as the original information being represented in complex-valued matrices, it is challenging to perform forward propagation within primarily real-valued neural networks. This is because complex-valued data and real-valued networks are inherently incompatible. Some manual feature transformation methods are either overly complex or do not yield satisfactory results. This paper proposes a hybrid model combining the advantages of CNNs and Transformers, termed the CFAT (Convolutional Fieldy Attention Transformer) model. This model aims to capture local features using CNNs while leveraging Transformers to capture global dependencies, thereby improving classification accuracy. CFAT utilizes simple full-real 9-D features as input, simplifying preprocessing steps while retaining important information from the original data. Experimental results on three commonly used PolSAR image datasets—Flevoland, San Francisco, and Oberpfaffenhofen—show that CFAT outperforms several baseline models, achieving classification accuracies of 99.28%, 98.50%, and 97.02%, respectively. This study demonstrates that integrating CNNs and Transformers offers significant advantages in PolSAR image classification tasks and provides new directions for future research.

## Usage

python main.py

## Requirements

torch>=1.12.1

torchvision>=0.13.1

torchaudio>=0.12.1

scikit-learn>=1.0.2

spectral>=0.24