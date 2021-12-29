import numpy as np
import cv2

def plot_colors(hist, centroids,percent_flag= True,width=600):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    height = int(width / 10)
    bar = np.zeros((height, width, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    hist_index= np.argsort(hist)[::-1]#reverse the list
    new = centroids[hist_index]
    hist=sorted(hist,reverse=True)
    if percent_flag:

        for (percent, color) in zip(hist, new):
            # plot the relative percentage of each cluster
            endX = startX + (percent * width)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), height),
                          color.astype("uint8").tolist(), -1)
            startX = endX
    else:
        step = width/len(new)
        for (percent, color) in zip(hist, new):
            # plot the relative percentage of each cluster
            endX = startX + step
            cv2.rectangle(bar, (int(startX), 0), (int(endX), height),
                          color.astype("uint8").tolist(), -1)
            startX = endX
    # return the bar chart
    return bar
