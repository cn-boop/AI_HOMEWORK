##Requirements
* python3.6
* pytorch
* tqdm
* space 2.0.11
* tensorboardX
* absl-py

##Usage
Download and preprocess the data
		
		#dowunload SQUAD2.0 and Glove
		$ sh download.sh
		$ python3.6 main.py --mode data

Train the model
		
		#model/model.pt will be generated every epoch
		$ python3.6 main.py --mode train
		
## Tensorboard
		
		$ tensorboard --logdir ./log/
## Score
F1=69.8 EM=66.1