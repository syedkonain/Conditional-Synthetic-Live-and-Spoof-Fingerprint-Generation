# ****Conditional Synthetic Live and Spoof Fingerprint Generation****

## DB2 Dataset

Download 1,500 synthetically generated live fingerprints along with their corresponding synthetic spoof fingerprints for 8 spoof materials across 10 finger classes (1–10). Each class contains 150 fingerprints per finger.

**Download link:**
https://drive.google.com/drive/folders/1_jGAdcCOXvoqpCEcPKae35O09A9EilrN

## DB3 Dataset

Download 1,500 synthetically generated live fingerprints together with the corresponding synthetic spoof fingerprints for 10 finger classes (1–10), with 150 fingerprints per finger.

**Download link:**
https://drive.google.com/drive/folders/1DS5lE8kRD24Idc-mpE-DQjBEf8gqZQmm

## Pretrained Model Weights

The trained weights for both models, along with the weights corresponding to all 8 spoof materials, are available at:

https://drive.google.com/drive/folders/1cTsB4OBuWT5M9M3mLV1GXHI3ti3QR1RX

## Usage

### DB2 Dataset Generation

1-Clone the official StyleGAN2-ADA repository.

2-Install the required dependencies.

3-Download the official CycleGAN repository inside the same StyleGAN2-ADA directory.

4-Install the CycleGAN dependencies.

5-Download our pretrained weights for DB2 live and all 8 spoof materials, and place them in the project folder.

6-Download the gen_db2.py script for image generation.

#### Generation Parameters

| Argument | Description |
|--------|------------|
| `--class` | Select the desired finger class (1–10) |
| `--name` | Choose `live` or a spoof material |
| `--num-impressions` | Number of impressions per fingerprint |
| `--num-images` | Total number of fingerprints to generate |
| `--seeds` | Specify seed range for controlled generation |

#### Spoof Material Names
- Live  
- EcoFlex  
- PlayDoh  
- Wood Glue  
- Gelatin  
- Latex  
- OOMOO  
- Silicone  
- Body Double 

#### Example Command
```bash
python gen_db2.py \
  --outdir DB2 \
  --network /home/DB2.pkl \
  --num-impressions 3 \
  --num-images 50 \
  --class 1 \
  --name live
```

### DB3 Dataset Generation

1-Clone the official StyleGAN3 repository.

2-Install the required dependencies.

3-Download the official CycleGAN repository inside the same StyleGAN3 directory.

4-Install the CycleGAN dependencies.

5-Download our pretrained weights for DB3 live and all 8 spoof materials, and place them in the project folder.

6-Download the gen_db3.py script for image generation.

#### Generation Parameters

| Argument | Description |
|--------|------------|
| `--class` | Select the desired finger class (1–10) |
| `--name` | Choose `live` or a spoof material |
| `--num-impressions` | Number of impressions per fingerprint |
| `--num-images` | Total number of fingerprints to generate |
| `--seeds` | Specify seed range for controlled generation |

#### Spoof Material Names
- Live  
- EcoFlex  
- PlayDoh  
- Wood Glue  
- Gelatin  
- Latex  
- OOMOO  
- Silicone  
- Body Double 

#### Example Command
```bash
python gen_db3.py \
  --outdir DB3 \
  --network /home/DB3.pkl \
  --num-impressions 3 \
  --num-images 50 \
  --class 1 \
  --name live
```
## Citation

If you use this dataset, pretrained models, or code in your research, please cite the following paper:

```bibtex
@article{https://doi.org/10.1049/bme2/7736489,
author = {Abbas, Syed Konain and Purnapatra, Sandip and Sarwar Murshed, M. G. and Miller-Lynch, Conor and Igene, Lambert and Dey, Soumyabrata and Schuckers, Stephanie and Hussain, Faraz},
title = {Conditional Synthetic Live and Spoof Fingerprint Generation},
journal = {IET Biometrics},
volume = {2026},
number = {1},
pages = {7736489},
keywords = {conditional GANs, presentation attack detection, synthetic fingerprint generation, synthetic spoof fingerprint generation},
doi = {https://doi.org/10.1049/bme2/7736489},
url = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/bme2/7736489},
eprint = {https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/bme2/7736489},
abstract = {Large fingerprint datasets, while important for training and evaluation, are time-consuming and expensive to collect and require strict privacy measures. Researchers are exploring the use of synthetic fingerprint data to address these issues. This article presents a novel approach for generating synthetic fingerprint images (both spoof and live), addressing concerns related to privacy, cost, and accessibility in biometric data collection. Our approach utilizes conditional StyleGAN2-ADA and StyleGAN3 architectures to produce high-resolution synthetic live fingerprints, conditioned on specific finger identities (thumb through little finger). Additionally, we employ CycleGANs to translate these into realistic spoof fingerprints, simulating a variety of presentation attack materials (e.g., EcoFlex, Play-Doh). These synthetic spoof fingerprints are crucial for developing robust spoof detection systems. Through these generative models, we created two synthetic datasets (DB2 and DB3), each containing 1500 fingerprint images of all 10 fingers with multiple impressions per finger, and including corresponding spoofs in eight material types. The results indicate robust performance: our StyleGAN3 model achieves a Fréchet inception distance (FID) as low as 5, and the generated fingerprints achieve a true acceptance rate (TAR) of 99.47\% at a 0.01\% false acceptance rate (FAR). The StyleGAN2-ADA model achieved a TAR of 98.67\% at the same 0.01\% FAR. We assess fingerprint quality using standard metrics (NFIQ2, MINDTCT), and notably, matching experiments confirm strong privacy preservation, with no significant evidence of identity leakage, confirming the strong privacy-preserving properties of our synthetic datasets.},
year = {2026}
}
```
## Acknowledgments

We utilized the official implementations of StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch), StyleGAN3 (https://github.com/NVlabs/stylegan3), and CycleGAN https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
