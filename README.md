# Conditional-Synthetic-Fingerprint-Image-Generation

**DB2 Dataset**:

Download 1,500 synthetically generated live fingerprints along with their corresponding synthetic spoof fingerprints for 8 spoof materials across 10 finger classes (1–10). Each class contains 150 fingerprints per finger.

Link: https://drive.google.com/drive/folders/1_jGAdcCOXvoqpCEcPKae35O09A9EilrN

**DB3 Dataset**:

Download 1,500 synthetically generated live fingerprints together with the corresponding synthetic spoof fingerprints for 10 finger classes (1–10), with 150 fingerprints per finger.

Link: https://drive.google.com/drive/folders/1DS5lE8kRD24Idc-mpE-DQjBEf8gqZQmm

The trained weights for both models, along with the weights corresponding to all 8 spoof materials, are available at:

https://drive.google.com/drive/folders/1cTsB4OBuWT5M9M3mLV1GXHI3ti3QR1RX

**USAGE:**

**DB2 Dataset Generation**

1-Clone the official StyleGAN2-ADA repository.

2-Install the required dependencies.

3-Download the official CycleGAN repository inside the same StyleGAN2-ADA directory.

4-Install the CycleGAN dependencies.

5-Download our pretrained weights for DB2 live and all 8 spoof materials, and place them in the project folder.

6-Download the gen_db2.py script for image generation.

**Generation Parameters (Fully Configurable)****
Argument	          Description
--class	            Select the desired finger class (1–10)
--name	            Choose live or a spoof material
--num-impressions	  Number of impressions per fingerprint
--num-images	      Total number of fingerprints to generate
--seeds	            Specify seed range for controlled generation

**Spoof Material Names**

Live

EcoFlex

PlayDoh

Wood Glue

Gelatine

Latex

OOMOO

Silicone

Body Double

**Example Command**
python gen_db2.py --outdir Db2 --network /home/DB2.pkl --num-impressions 3 --num-images 50 --class 1 --name live


**DB3 Dataset Generation**:

1-Clone the official StyleGAN3 repository.

2-Install the required dependencies.

3-Download the official CycleGAN repository inside the same StyleGAN2-ADA directory.

4-Install the CycleGAN dependencies.

5-Download our pretrained weights for DB3 live and all 8 spoof materials, and place them in the project folder.

6-Download the gen_db3.py script for image generation.

**Generation Parameters**

Argument	          Description
--class	            Select the desired finger class (1–10)
--name	            Choose live or a spoof material
--num-impressions	  Number of impressions per fingerprint
--num-images	      Total number of fingerprints to generate
--seeds	            Specify seed range for controlled generation

**Spoof Material Names**
Live

EcoFlex

PlayDoh

Wood Glue

Gelatine

Latex

OOMOO

Silicone

Body Double

**Example Command**

python gen_db3.py --outdir Db3 --network /home/DB3.pkl --num-impressions 3 --num-images 50 --class 1 --name live


These resources are available upon request; interested researchers should email syedkonainabas@gmail.com to obtain access.


**Acknowledgments**:
We utilized the official implementations of StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch), StyleGAN3 (https://github.com/NVlabs/stylegan3), and CycleGAN https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
