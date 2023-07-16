#import libraries
import os
import PIL
import cv2 
import glob 
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab

#load model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')

def clear_widget():
    global cv 
    #to clear canvas
    cv.delete("all")

def activate_event(event):
    global lastx, lasty
    # <B1-Motion>
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # do the canvas drawing
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    lastx, lasty = x, y 

def Recognize_Digit():
    global root, cv
    filename = f'image.png'


    #grab the image, crop it according to my requirement and saved it in png format
    ImageGrab.grab().crop((0,40,1000,750)).save(filename)

    #read the image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    #convert the image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #applying Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #findContour() function helps in extracting the contours from the image.
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        #get bounding box and extract ROI
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
        top = int(0.005 * th.shape[0])
        bottom = top
        left = int(0.005 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)

        #Extract the ROI
        roi = th[y-top: y+h+bottom, x-left:x+w+right]

        #resize and reshape roi image
        img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1,28,28,1)

        #normalizing the image to support our model input
        img = img/255.0

        #prediction
        pred = model.predict(img)[0]

        final_pred = np.argmax(pred)
        data = str(final_pred)#+' '+str(int(max(pred))*100)+ '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thickness = 1
        cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)

    #Showing the predicted results on new window
    image = cv2.resize(image, (600, 500))
    cv2.imshow('image', image)
    cv2.waitKey(0)


#create a main window first (name as root)
root = Tk()
# root.resizable(0,0)
root.geometry('650x520')
root.title('Handwritten Digit Recognition')

#Initialize few variables
lastx, lasty = None, None
image_number = 0

#Create a canvas for drawing
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

#Tkinter provides a powerful mechanism to let you deal with events yourself.
cv.bind('<Button-1>', activate_event)

#Add buttons and labels
btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Clear Widget', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
root.state('zoomed')

#mainloop() is used when your application is ready to run.
root.mainloop()

