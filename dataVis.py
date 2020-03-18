import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as anim

def create_animation(buyerJoints, leftSellerJoints, rightSellerJoints, fileName):
    """
    Generates a video file of the subjects' motion
    :param buyerJoints: 'joints19' of buyer
    :param leftSellerJoints: 'joints19' of left seller
    :param rightSellerJoints: 'joints19' of right seller
    :param fileName: fileName of the output animation
    :return: None
    """

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

    # add the joints and the marker colors
    # setup the frame layout
    fig = plt.figure()
    plt.axis('off')
    ax = p3.Axes3D(fig)
    ax.view_init(elev=-60, azim=-90)
    ax._axis3don = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set up the writer
    # Set up formatting for the movie files
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Kurel002'), bitrate=1800)

    jointList = []
    leftSellerPoints = []
    for i in range(0, len(leftSellerJoints), 3):
        x = leftSellerJoints[i][0]
        y = leftSellerJoints[i + 1][0]
        z = leftSellerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='r', marker="o")
        leftSellerPoints.append(graph)

    leftSellerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='r')
        leftSellerBones.append(graph)

    jointList = []
    rightSellerPoints = []
    for i in range(0, len(rightSellerJoints), 3):
        x = rightSellerJoints[i][0]
        y = rightSellerJoints[i + 1][0]
        z = rightSellerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='b', marker="o")
        rightSellerPoints.append(graph)

    rightSellerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='b')
        rightSellerBones.append(graph)

    jointList = []
    buyerPoints = []
    for i in range(0, len(buyerJoints), 3):
        x = buyerJoints[i][0]
        y = buyerJoints[i + 1][0]
        z = buyerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='g', marker="o")
        buyerPoints.append(graph)

    buyerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='g')
        buyerBones.append(graph)

    def update_plot(frame, joints, points, bones, humanSkeleton):

        leftSellerPoints = points[0]
        rightSellerPoints = points[1]
        buyerPoints = points[2]

        leftSellerJoints = joints[0]
        rightSellerJoints = joints[1]
        buyerJoints = joints[2]

        leftSellerBones = bones[0]
        rightSellerBones = bones[1]
        buyerBones = bones[2]

        jointList = []
        for i in range(len(leftSellerPoints)):
            graph = leftSellerPoints[i]
            x = leftSellerJoints[3 * i][frame]
            y = leftSellerJoints[3 * i + 1][frame]
            z = leftSellerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(leftSellerBones)):
            start, end = humanSkeleton[i]
            graph = leftSellerBones[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

        jointList = []
        for i in range(len(rightSellerPoints)):
            graph = rightSellerPoints[i]
            x = rightSellerJoints[3 * i][frame]
            y = rightSellerJoints[3 * i + 1][frame]
            z = rightSellerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(rightSellerBones)):
            graph = rightSellerBones[i]
            start, end = humanSkeleton[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

        jointList = []
        for i in range(len(buyerPoints)):
            graph = buyerPoints[i]
            x = buyerJoints[3 * i][frame]
            y = buyerJoints[3 * i + 1][frame]
            z = buyerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(buyerBones)):
            graph = buyerBones[i]
            start, end = humanSkeleton[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

    points = [leftSellerPoints, rightSellerPoints, buyerPoints]
    joints = [leftSellerJoints, rightSellerJoints, buyerJoints]
    bones = [leftSellerBones, rightSellerBones, buyerBones]
    frames = len(leftSellerJoints[0])

    ani = anim.FuncAnimation(fig, update_plot, frames, fargs=(joints, points, bones, humanSkeleton), interval=50, repeat=False)

    # save only if filename is passed
    if fileName:
        print(fileName+'.mp4')
        ani.save(fileName + '.mp4', writer=writer)
    else:
        plt.show()

    # close plot
    plt.close()




