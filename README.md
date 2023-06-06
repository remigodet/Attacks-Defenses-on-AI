# Attacks-Defenses-on-AI

Project on studying a corpus of physical adversarial attacks on stop signs.

Made for the AI project class of CentraleSupelec.
___

## Attacks studied

### Shapeshifter

*Rémi Godet*

For shapeshiter follow these steps :

1. install the environment described in Shapeshifter/requirements_shapefiter.txt
If there is trouble with the object detection librabry you can :

- Use the command :

```bash

pip install -e "git+https://github.com/tensorflow/models.git#egg=object_detection&subdirectory=research/object_detection/packages/tf1&ref=fe748d4a4a1576b57c279014ac0ceb47344399c4&editable=true"
```

- Clone the code from <https://github.com/tensorflow/models/tree/master/research/object_detection>

2. Create the perturbations using the notebook in Shapeshifter folder (skip the "Compute model performance on images" part and choose your targeted labels in the intro variables )

3. Take pictures !

4. Infer on the pictures following the notebook in Shapeshifter
5. Perform data cleaning with visualisation.py
6. You now have the final Excel for insights and visualisation

    *I recommend using dynamic tables in Excel, see the final_results.xlsx in this repo.*

Original work : <https://github.com/shangtse/robust-physical-attack>
___

### Patch attack

*Benjamin Rio*
Install requirements with the `requirements.txt` file in the patch_attack folder.
The folder contains the code for training the classifier trained on GTRSB dataset, the code for generating the patch and the code for attacking the stop sign.
The training dataset of the classifier can be found [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
___

### GRAPHITE

*Elian Mangin*

### Packages

```
matplotlib==3.1.2
numpy==1.17.4
prettytable==2.2.0
opencv_python==4.2.0.32
pandas==0.25.3
torchvision==0.8.1
joblib==0.14.0
tqdm==4.60.0
scipy==1.3.3
torch==1.7.0
kornia==0.5.10
Pillow==9.0.0
python_Levenshtein==0.12.2
```

### Setup

First off, download the GTSRB dataset and to put it in the same folder as GRAPHITE/

### General usage

```
python3 main.py -v <victim class index> -t <target class index> --tr_lo <tr_lo> --tr_hi <tr_hi> -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```

### Running GTSRB attacks from Table 8

Stop sign to Speed Limit 30: <br>

```
python3 main.py -v 14 -t 1 --tr_lo 0.65 --tr_hi 0.85 -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```

Stop sign to Pedestrians: <br>

```
python3 main.py -v 14 -t 27 --tr_lo 0.65 --tr_hi 0.85 -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```

Example outputs from Table 8 included in `example_outputs`.

Original work : <https://github.com/ryan-feng/GRAPHITE>
___

### Laser Attack

*Camille Lançon*

### Packages

```
cv2==4.6.0
numpy==1.23.4
matplotlib==3.6.0
PIL==9.2.0
tensorflow==2.12.0
keras==2.12.0
sklearn==1.1.3
```

For Laser Attack, follow these steps :

1. Download the GHTSRB dataset at this address : <https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>
2. Name the dataset "GHTSRB" and put it in the folder LaserAttack.
3. Run the notebook "train_model".
4. Run the notebook "laser_attack". This attack is performed on the image called "stop.png". You can replace this image by another one if you want to perform the attack on another stop sign.
5. The parameters of the laser beam are given at the end of the notebook.
6. You can watch the results in the folder "Results".

___

### Adversarial texture attack

*Lamiaa El Bouchattaoui*
___
