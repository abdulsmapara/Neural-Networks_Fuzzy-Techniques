import cv2



# Compute the frame difference
def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return diff_frames1 & diff_frames2



print(cv2.__version__)
vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0
success = True
optimal_threshold = 100

background = []
foreground = []

gray = image
prev = image
while success and count < 10:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1