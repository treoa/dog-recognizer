# Dog Recognizer

### For testing on images:

1. Clone thir repository
2. Navigating into the folder, containing this repo, clone the repository of darknet, using
```
git clone https://github.com/AlexeyAB/darknet
```

or by launching ```test.py ``` file and running the ```clone_repo``` function

3. Configure Makefile as in the second cloned repository, or by launching ```test.py ``` file and running the ```make_file``` function
4. after running ```make``` manually or through ```test.py```, ```darknet.sh``` script must be in the root darknet folder
5. Download pre-trained yolov4 weights
6. Define helper functions of showing a prediction. It is given in the ```test.py``` file indicating function ```imShow```.