# Face-Recognition

1. create your own dataset by runnig the data_create.py file and bringing your face in front of the camera.
2. move the images created in Images folder to Datasets subfolder (structure mentioned below)
       /Datasets
       -------/person1/
       -------/person2/
3. run the file Model.py. You can change your paramters as you wish. Once your model is saved, the weights will be saved in the .h5 file will be saved.
4. add the path of this .h5 file in your model=load_model() variable in detector.py file.
