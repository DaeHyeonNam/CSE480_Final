# CSE480_Final


## Assignment 2-2 (Gesture recognition - rock, scissors,and paper)
![gesture_recog_demo](./demo/demo.gif)

### Installation
~~~
pip install keyboard, torch, mediapipe, opencv-python
~~~

### Data Collection
~~~
python dataCollection.py
~~~
Press 'a' to save the coordinates of landmarks in data.csv file
![data_collection](./demo/data_collecting.gif)

### Data Processing
Run 'data processing.ipynb' file to generate train and validation data

### train
Run 'train.ipynb' file to train! 
But you can also use the pretrained weights in weight folder. (the code for using pretrained-weight is in the last cell of 'train.ipynb' file)

### Real-time Run!
~~~
python realtime_classification.py
~~~
Since this file is using pretrained weights, you don't need above processes to run it!
