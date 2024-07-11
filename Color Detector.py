#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install opencv-python


# In[2]:


import cv2
import numpy as np


# In[3]:


color_ranges = {
    "red1": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "green": ((36, 25, 25), (86, 255, 255)),
    "blue": ((94, 80, 2), (126, 255, 255)),
    "yellow": ((15, 150, 150), (35, 255, 255)),
    "orange": ((10, 100, 20), (25, 255, 255)),
    "purple": ((130, 50, 50), (160, 255, 255)),
}


# In[ ]:


def detect_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                return color
    return "Unknown"


# In[ ]:


cap = cv2.VideoCapture(0)


# In[ ]:


while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = detect_color(frame)
    cv2.putText(frame, f"Color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

