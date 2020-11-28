# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


#Read a video from webcam (Frame by Frame)
#Each frame is an image and video is a series of frames
import cv2
import numpy as np
#Capturing a device
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person : ")


while True :
	ret,frame=cap.read()   #returns a bool value and a frame
	#For gray color
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	

	if ret==False :
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)  #Scaling factor and number of neighbours
	# Each face is a list of tuples : x,y,w,h
	if len(faces)==0:
		continue
		
	faces=sorted(faces,key=lambda f : f[2]*f[3])  #Sorted the faces


	#Drawing a rectangle around face
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))

	for face in faces[-1:] :
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))



	cv2.imshow("Video Frame",frame)
	cv2.imshow("Face Section",face_section)
	#cv2.imshow("Gray Frame",gray_frame)

	#Wait for user-input and break the loop when key is equal to q
	#cv2.waitkey returns 32 bit integer and doing & with 0xff(8 bit integer equal to 1)
	#returns an 8 bit integer suitable for comparison with ascii value 
	key_pressed=cv2.waitKey(1) & 0xFF

	if(key_pressed==ord('q')) :    #ord returns ASCII value of char
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at "+dataset_path+file_name+'.npy')


cap.release()
cv2.destroyAllWindows()		





