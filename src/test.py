import os
import cv2
import zipfile
import matplotlib.pyplot as plt
import time
import generate_test, generate_train


def clone_repo():
	os.system('git clone https://github.com/AlexeyAB/darknet')


def make_file():
	os.system('''sed -i 's/OPENCV=0/OPENCV=1/' Makefile''')
	os.system('''sed -i 's/GPU=0/GPU=1/' Makefile''')
	os.system('''sed -i 's/CUDNN=0/CUDNN=1/' Makefile''')
	os.system('''sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile''')
	print(f"DONE CHANGING MAKEFILE FOR GPU AND OPENCV ENABLES\n\n")
	os.system('/usr/local/cuda/bin/nvcc --version')
	print(f"STARTING MAKING MAKEFULE\n\n")
	os.system('make')
	print(f"DONE MAKING MAKEFILE\n\n")


def download_weights():
	os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights')
	print(os.getcwd())


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


def unzip_data ():
	print(f"{os.getcwd()} - is where unzipping processed\n\n")
	os.chdir('darknet') if not os.getcwd().endswith('darknet') else print("I am in the darknet folder\n\n")
	os.system('unzip ../data/train.zip -d data/')
	os.system('unzip ../data/test.zip -d data/')


def test_1 ():
	print(os.getcwd() + ' is where the first test is being processed\n')
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

	


def train_continue ():
	print(os.getcwd() + ' is where the kicking off from last saved training continues\n\n')
	# kick off training from where it last saved
	os.system('./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_last.weights -dont_show')


def train_stats ():
	print(os.getcwd() + ' is where the stats of the training is displayed')
	# show chart.png of how custom object detector did with training
	imShow('chart.png')
	# weights needed to be changed for displaying the mean average performance of the model and needed to be run in darknet folder
	os.system('./darknet detector map data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_5000.weights')


def test_custom ():
	print(os.getcwd() + ' is where the third test (on real data and custom model) is being processed\n\n')
	# needed to be performed in cfg folder in darknet
	os.system('''sed -i 's/batch=64/batch=1/' yolov4-obj.cfg''')
	os.system('''sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg''')
	# here again needed to be moved to the folder back for the next command
	os.chdir('darknet') # Needed for testing separately
	print(os.getcwd()+ '\n current directory which I am working in \n')
	os.system('./darknet detector test obj.data yolov4-obj.cfg backup/yolov4-obj_last.weights 293566393jpg -thresh 0.3')


if __name__ == "__main__":
	print(f"\n\nThe current directory is {os.getcwd()}\n\n")
	if not os.path.exists('darknet'):
		clone_repo()
		print(f"Done cloning the repo\n\n")
	os.chdir(os.getcwd() + '/darknet')
	if not os.path.exists('darknet'):
		make_file()
		print(f"Done making file\n\n")
	if not os.path.exists('yolov4.weights'):
		download_weights()
		print(f"Done downloading the weights\n\n")
	print(f"Now the stage of preparing the data\n\n")
	unzip_data()
	print(f"Here is the step of configuring obj.names, obj.data and .cfg file from cfg directory (from cfg directory only yolov4-custom.cfg file and save it to cfg folder after configuring as yolov4-obj.cfg)\n\n")
	print(f"If you have missed the file yolov4-obj.cfg in your root directory of darknet, please terminate current run and configure it. After what run again. Every made change will be kept\n\n")
	time.sleep(10)
	os.system('cp yolov4-obj.cfg cfg/yolov4-obj.cfg')
	print(f"Now the copying is done\n\n")
	print(f"Defining train and test generation functions")
	os.chdir("../src")
	os.system("cp generate_test.py ../darknet")
	os.system("cp generate_train.py ../darknet")
	os.chdir("../data")
	os.system("cp obj.names ../darknet/data")
	os.system("cp obj.data ../darknet/data")
	os.chdir("../darknet")
	os.system("python3 generate_train.py")
	os.system("python3 generate_test.py")
	print(f"Generating data for test and train has been finished and now I am on {os.getcwd()}\n\n")
	print(f"Downloading pre-trained weights for convolutional layer\n\n")
	os.system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')
	options = ["Start training the model", "Continue from last started", "Show the training statistics", "Start testing the model"]
	for i in range(len(options)):
		print(f"[{str(i+1)}] - {options[i]}")
		selected = False;
	while not selected:
		inp = int(input("Enter a number: "))
		if inp in range(1, len(options)+1):
		    inp = options[inp-1]
		else:
		    print("Invalid input!")
	if inp == 1:
		print(f"I am starting training the model... \n")
		# training of custom detector (uncomment %%capture below if code is run into memory issues)
		# %%capture	
		os.system('./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map')

