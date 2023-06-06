# Attacks-Defenses-on-AI

Project on studying a corpus if physical adversarial attacks on stop signs. 

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

- Clone the code from https://github.com/tensorflow/models/tree/master/research/object_detection

2. Create the perturbations using the notebook in Shapeshifter folder (skip the "Compute model performance on images" part and choose your targeted labels in the intro variables )

3. Take pictures ! 

4. Infer on the pictures following the notebook in Shapeshifter
5. Perform data cleaning with visualisation.py
6. You now have the final Excel for insights and visualisation 

    *I recommend using dynamic tables in Excel, see the final_results.xlsx in this repo.*

___
### Patch attack
*Benjamin Rio*
___
### GRAPHITE
*Elian Mangin*
___
### Laser Attack
*Camille Lançon*
___


