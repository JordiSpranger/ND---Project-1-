import numpy as np
import cv2
import copy

# import video file
cap = cv2.VideoCapture('challenge.mp4')
ret, frame = cap.read()

# video dimentions
xsize = frame.shape[1]
ysize = frame.shape[0]
print("video dimentions :)","X-pixels:",xsize,"Y-pixels:", ysize)

#triangle of interest
left_bottom = [0, ysize]
right_bottom = [xsize, ysize]
apex = [xsize*0.5, ysize*0.5]

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 180


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 5
max_line_gap = 2


#while the video is playing
while(cap.isOpened()):
    #return frame
    ret, frame = cap.read()

    #make a copy of original frame
    original = copy.copy(frame)

    if str(type(frame)) == "<type 'numpy.ndarray'>":

        #gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.blur(frame, (5, 5))
        frame = cv2.blur(frame, (5, 5))

        #erode & dilute
        frame = cv2.erode(frame, (20,20))
        frame = cv2.dilate(frame, (15,15))
        #edges
        frame = cv2.Canny(frame, 120, 150)


        #mask
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array([[(xsize*0.2, ysize*0.9), (xsize*0.4, ysize*0.7),(xsize*0.8, ysize*0.70), (xsize*0.85, ysize*0.9)]], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        #channel_count = frame.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255)
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex

        # apply the mask
        masked_image = cv2.bitwise_and(frame, mask)

        #
        edges = cv2.Canny(masked_image, low_threshold, high_threshold)

        # Run Hough on edge detected image
        line_image = np.copy(original) * 0 # creating a blank to draw lines on

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([255,0,0]),
        min_line_length, max_line_gap)


        # Iterate over the output "lines" and draw lines on the blank
        if str(type(lines)) == "<type 'numpy.ndarray'>":
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 20)


        # Draw the lines on the edge image
        combo = cv2.addWeighted(line_image, 1, original, 1, 0)


        cv2.imshow('frame',combo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else :
        break

cap.release()
cv2.destroyAllWindows()