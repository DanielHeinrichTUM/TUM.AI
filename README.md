# TUM.AI
 AI Projects, swiftly improvised for a short-ish notice application.
 I worked on the occular disease detection challange from Kaggle: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?resource=download
 Started work on the 20.04. and had first result worth mentioning at the 23.04. so I am quite happy to have managed to scrap something together so quickly.

 The Ocular Disease Intelligent Recognition (ODIR) dataset contains 5,000 patient records, each consisting of:

    1. Left and right eye fundus photographs (captured using various cameras, including Canon, Zeiss, and Kowa)
    2. Patient age
    3. patient gender

The Problem essentially breaks down to a fairly easy and straight forward multi-class classification challange, with the classes being:

    1. N - Normal
    2. D - Diabetes
    3. G - Glaucoma
    4. C - Cataract
    5. A - Age-related Macular Degeneration
    6. H - Hypertension
    7. M - Pathological Myopia
    8. O - Other diseases/abnormalities

My initial plan right out of the gate was as follows:

    1. Use transfer learning with a pretrained CV model (like ResNet50, which is what I ended up using)
    2. Unfreeze the last layer and add some new FC layers at the end to also consider age and gender as additional parameters
    3. Final layer for the 8 class-outputs
    4. Train with the dataset
    5. ....
    6. Profit?

Edit: Very first training run already resulted in a model with 65% val accuracy which is quite neat. 
However I have had some more considerations in the mean time:

First of all Dataset is very unbalanced -> next I will look into a combination of oversampling and data augmentation to boost the underrepresented classes during training.
Also, it is time to add some more quality of live code like closer tracking of train and val loss during training, instead of just getting feedback at the end of every epoch. And while at it, I added intermediate model saving after every epoch and upon interruption of the training.