import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FFMpegWriter
import io
import cv2
import numpy as np

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=800):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# create the connection list for skeleton
humanSkeleton = [
    [0,1],
    [0,3],
    [3,4],
    [4,5],
    [0,2],
    [2,6],
    [6,7],
    [7,8],
    [2,12],
    [12,13],
    [13,14],
    [0,9],
    [9,10],
    [10,11]
]


# load the data
file = 'test_file.pkl'
file = open(file, 'rb')
group = pickle.load(file)

# load buyer, seller Ids
leftSellerId = group['leftSellerId']
rightSellerId = group['rightSellerId']
buyerId = group['buyerId']

# load skeletons
buyerJoints = []
leftSellerJoints = []
rightSellerJoints = []

# load the skeletons
for subject in group['subjects']:
    if(subject['humanId'] == leftSellerId):
        leftSellerJoints = subject['joints19']
    if (subject['humanId'] == rightSellerId):
        rightSellerJoints = subject['joints19']
    if (subject['humanId'] == buyerId):
        buyerJoints = subject['joints19']

frames = []
for frame in range(100, 200, 10):
    # add the joints and the marker colors
    # setup the frame layout
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(elev=-60, azim=-90)

    jointList = []
    for i in range(0, len(leftSellerJoints), 3):
        x = leftSellerJoints[i][frame]
        y = leftSellerJoints[i+1][frame]
        z = leftSellerJoints[i+2][frame]
        jointList.append((x, y, z))
        ax.scatter(x, y, z, c='r')

    for start,end in humanSkeleton:
        ax.plot([jointList[start][0], jointList[end][0]], [jointList[start][1], jointList[end][1]], [jointList[start][2], jointList[end][2]], c='r')

    jointList = []
    for i in range(0, len(rightSellerJoints), 3):
        x = rightSellerJoints[i][frame]
        y = rightSellerJoints[i+1][frame]
        z = rightSellerJoints[i+2][frame]
        jointList.append((x, y, z))
        ax.scatter(x, y, z, c='b')

    for start,end in humanSkeleton:
        ax.plot([jointList[start][0], jointList[end][0]], [jointList[start][1], jointList[end][1]], [jointList[start][2], jointList[end][2]], c='b')

    jointList = []
    for i in range(0, len(buyerJoints), 3):
        x = buyerJoints[i][frame]
        y = buyerJoints[i+1][frame]
        z = buyerJoints[i+2][frame]
        jointList.append((x, y, z))
        ax.scatter(x, y, z, c='g')

    for start,end in humanSkeleton:
        ax.plot([jointList[start][0], jointList[end][0]], [jointList[start][1], jointList[end][1]], [jointList[start][2], jointList[end][2]], c='g')

    plt.show()
    frames.append(get_img_from_fig(fig))

