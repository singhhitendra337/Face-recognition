#Read a video from webcam (Frame by Frame)
#Each frame is an image and video is a series of frames
import cv2
#Capturing a device
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True :
	ret,frame=cap.read()   #returns a bool value and a frame
	#For gray color
	#gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False :
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)  #Scaling factor and number of neighbours
	

	for (x,y,w,h) in faces :
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  #BGR values

	cv2.imshow("Video Frame",frame)
	#cv2.imshow("Gray Frame",gray_frame)

	#Wait for user-input and break the loop when key is equal to q
	#cv2.waitkey returns 32 bit integer and doing & with 0xff(8 bit integer equal to 1)
	#returns an 8 bit integer suitable for comparison with ascii value 
	key_pressed=cv2.waitKey(1) & 0xFF

	if(key_pressed==ord('q')) :    #ord returns ASCII value of char
		break

cap.release()
cv2.destroyAllWindows()		



"""
scaleFactor – Parameter specifying how much the image size is reduced at each image scale.

Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.



minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.

This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

"""


