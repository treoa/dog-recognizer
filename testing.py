import time

detections = False

detect_num = 0
while True:
	detect_num += 1 if detections else 0
	print(detect_num)
	time.sleep(2)
	