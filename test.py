import os
import cv2
import matplotlib.pyplot as plt


# [x] Clone git repo- darknet to the current directory
# [x] go to darknet folder, which was cloned before
# [x] configure makefile and run makefile
# [x] darknet.sh script must appear in the current directory folder after making it up
# [] download pre-trained yolov4 weights to the darknet folder
# [] define helper function of showing picture (im_Show)
# [] run test detection of darknet in the current darknet directory
# [] Prepare train and test datasets. Upload and convrt to yolov4 format before that
# [] unzip both train and test datasets to data folder in the darknet directory
#			unzip ../train.zip -d data/ (do it in the darknet folder)
# [] If no cfg file - copy it to custom place from cfg folder and configure it there
# [] Configure obj.names and obj.data files navigating properly (later in steps it will be known where the file is for configuring its directories)
# [] Copy configured cfg file back to the cfg folder in the darknet root directory
# [] Copy obj.names and obj.data files to the data folder in the darknet root directory
# [] define helper functions (generate_train and generate_test. And actually they are already given)
# [] Copy helper functions to the root directory of darknet
# [] run helper functions to generate two txt files in the root directory
# [] Donwload pre-trained weights for convolutional layer training
# [] Run training with these weights
#		./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont-show -map
# 		-dont-show flag is needed for not displaying the graph, because it could crash
#		If training crashes because of memory error, try to run before the training "capture" terminal command
# [] Define training for continuing, if previous training was somehow stopped, just changing the weights
# [] Check graph and mean average Precision (mAP) of the model
# [] Test trained model with last weights
# 		./darknet detector test data/obj.data cfg/yolov4-obj.cfg ../ackup/last.weights /path/to/the/image/test/on -thresh 0.35


# clone darknet
def clone_repo():
	os.system('git clone https://github.com/AlexeyAB/darknet')
	os.chdir(os.getcwd() + '/darknet')


def make_file():
	os.system('''sed -i 's/OPENCV=0/OPENCV=1/' Makefile''')
	os.system('''sed -i 's/GPU=0/GPU=1/' Makefile''')
	os.system('''sed -i 's/CUDNN=0/CUDNN=1/' Makefile''')
	os.system('''sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile''')
	print("\nDONE CHANGING MAKEFILE FOR GPU AND OPENCV ENABLES\n")
	os.system('/usr/local/cuda/bin/nvcc --version')
	print("\nSTARTING MAKING MAKEFULE\n")
	os.system('make')
	print("\nDONE MAKING MAKEFILE\n")


def download_weights():
	os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights')
	print(os.getcwd())

# image show
def imShow(path):
	# path = os.getcwd() + '/' + path
	print(os.getcwd()+" in the image show fn \n")
	image = cv2.imread(path)
	print(image)
	height, width = image.shape[:2]
	resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

	fig = plt.gcf()
	fig.set_size_inches(18, 10)
	plt.axis("off")
	plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
	plt.show()

	# NEEDED TO BE DONE FOR MODEL TRAINING
def unzip_data ():
	print(os.getcwd() + ' - is where unzipping processed\n')
	# unzip the datasets and their contents so that they are now in /darknet/data/ folder
	os.system('unzip ../train.zip -d data/')
	os.system('unzip ../test.zip -d data/')


def test_1 ():
	print(os.getcwd() + ' is where the first test is being processed\n')
	# os.chdir('darknet')
	os.system('./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /data/person.jpg')
	imShow('predictions.jpg')


def test_2 ():
	print(os.getcwd() + ' is where the second test is being processed\n')
	os.system('./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights ~/Projects/dog-recognizer/street_test.jpg')
	imShow('predictions.jpg')


def train_start ():
	print(os.getcwd() + ' is where the third test (on real data) is being processed\n')
	unzip_data()
	os.path.normpath(os.getcwd() + os.sep + os.pardir)
	print(os.getcwd() + ' is after moving back\n')
	os.system('python3 generate_train.py')
	os.system('python3 generate_test.py')
	os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')
	# train your custom detector! (uncomment %%capture below if you run into memory issues or your Colab is crashing)
	# %%capture
	os.system('./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map')


def train_continue ():
	print(os.getcwd() + ' is where the kicking off from last saved training continues\n')
	os.system('python3 generate_train.py')
	os.system('python3 generate_test.py')
	# kick off training from where it last saved
	os.system('./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show')


def train_stats ():
	print(os.getcwd() + ' is where the stats of the training is displayed')
	# show chart.png of how custom object detector did with training
	imShow('chart.png')
	# weights needed to be changed for displaying the mean average performance of the model and needed to be run in darknet folder
	os.system('./darknet detector map data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_5000.weights')


def test_custom ():
	print(os.getcwd() + ' is where the third test (on real data and custom model) is being processed\n')
	# needed to be performed in cfg folder in darknet
	os.system('''sed -i 's/batch=64/batch=1/' yolov4-obj.cfg''')
	os.system('''sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg''')
	# here again needed to be moved to the folder back for the next command
	os.chdir('darknet') # Needed for testing separately
	print(os.getcwd()+ '\n current directory which I am working in \n')
	os.system('./darknet detector test obj.data yolov4-obj.cfg backup/yolov4-obj_last.weights 293566393jpg -thresh 0.3')


# clone_repo()
# make_file()
# download_weights()
test_custom()