#from __future__ import print_function, absolute_import, division

import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from scipy.spatial.transform import Rotation

pos = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]

I   = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
J   = np.array([2,3,4,7,8,9,13,14,18,16,19,20,26,27,28])-1 # end points
LR  = np.array([1,1,1,0,0,0,0, 0,0, 0, 0, 0, 1, 1, 1], dtype=bool)

skeleton_mpii = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [6, 3], [12, 11], [8, 12], \
                 [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [8, 13]]

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_3d_keypoints(vals, ax, lcolor='red', rcolor='blue', elev=0, azim=0):
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[pos.index(I[i]), j], vals[pos.index(J[i]), j]] ) for j in range(3)]
        # x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor, zdir='z')

    RADIUS = 2 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    # Get rid of the ticks and tick labels
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    #ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    ax.w_zaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

def show_3d_prediction(pred, gt, show=False, azim=0, elev=0, pcolor="#3498db", gtcolor="#e74c3c", name='test'): # blue, orange
    axx = []
    ax = plt.subplot(121, projection='3d')
    axx.append(ax)

    x, y, z = [np.array( [pred[pos.index(I[0]), j], pred[pos.index(J[0]), j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=pcolor, zdir='z', label='pred')

    for i in np.arange(1, len(I)):
        x, y, z = [np.array( [pred[pos.index(I[i]), j], pred[pos.index(J[i]), j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=pcolor, zdir='z')

    plt.legend()

    ax = plt.subplot(122, projection='3d')
    axx.append(ax)

    x, y, z = [np.array( [gt[pos.index(I[0]), j], gt[pos.index(J[0]), j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=gtcolor, zdir='z', label='ground truth')

    for i in np.arange(1, len(I)):
        x, y, z = [np.array( [gt[pos.index(I[i]), j], gt[pos.index(J[i]), j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=gtcolor, zdir='z')

    plt.legend()

    RADIUS = 1.7 # space around the subject

    root = [pred[0,:],gt[0,:]]

    for j, ax in enumerate(axx):
        xroot, yroot, zroot = root[j]
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

        # Get rid of the ticks and tick labels
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        #ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        ax.w_zaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

    if show:
        plt.show()
    else:
        plt.savefig('output/' + name + '_3d.png')
    plt.close()

def show_3D_pose(vals, angles, keypoints_2d, img, show=False, azim=0, elev=0, name='test', lcolor="#3498db", rcolor="#e74c3c"): # blue, orange
    # ax = plt.subplot(111, projection='3d')
    if 'lank' in angles[1]:
        # rotations=[('x',[-90]), ('y',[90])] h36m
        rotations=[('z',[90]), ('y',[90])]
    else:
        #rotations=[('z', [90]), ('y',[-90])] h36m
        rotations=[('x', [90]), ('y',[-90])]
    fig = plt.figure()

    # 2D plot
    ax0 = fig.add_subplot(221)
    ax0.imshow(img)
    ax0.axis('off')
    ax0.scatter(keypoints_2d[:,0], keypoints_2d[:,1], color='cyan', marker='.')
    ax0.set_title("2D Detection")
    for l in skeleton_mpii:
        p1, p2 = l
        x = [keypoints_2d[p1,0], keypoints_2d[p2,0]]
        y = [keypoints_2d[p1,1], keypoints_2d[p2,1]]
        ax0.plot(x,y, color='cyan')

    # 3D plot (viewpoint #1)
    ax = fig.add_subplot(222, projection='3d')
    ax.set_title("3D Detection (Viewpoint #1)")
    plot_3d_keypoints(vals, ax, lcolor, rcolor, elev, azim)

    # 3D plot (viewpoint #2)
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.set_title("3D Detection (Viewpoint #2)")
    r = Rotation.from_euler(rotations[0][0], rotations[0][1], degrees=True)
    vals_view2 = r.apply(vals)
    plot_3d_keypoints(vals_view2, ax2, lcolor, rcolor, elev, azim)

    # 3D plot (viewpoint #3)
    ax3 = fig.add_subplot(224, projection='3d')
    ax3.set_title("3D Detection (Viewpoint #3)")
    r = Rotation.from_euler(rotations[1][0], rotations[1][1], degrees=True)
    vals_view3 = r.apply(vals)
    plot_3d_keypoints(vals_view3, ax3, lcolor, rcolor, elev, azim)

    # Add text displaying the angle for the selected exercise
    ax2.text2D(-0.2, -1.4, angles[1] + str(round(angles[0], 2)) + "??",
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox={'facecolor': 'black', 'alpha': 0.1, 'pad': 10},
            transform=ax.transAxes)

    if show:
        plt.show()
    else:
        plt.savefig('test.png', dpi=400)
        return get_img_from_fig(fig)
    plt.close()


def show_2D_data(keypoints, norm=True):

    plt.scatter(keypoints[:,0], keypoints[:,1], color='cyan', marker='o')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()
    # GT lines
    for l in skeleton_mpii:
        p1, p2 = l
        x = [keypoints[p1,0],keypoints[p2,0]]
        y = [keypoints[p1,1],keypoints[p2,1]]
        #plt.plot(x,y, color='gold')

    for i in range(keypoints.shape[0]):
        plt.text(keypoints[i,0], keypoints[i,1], str(i))
    
    if norm:
        plt.xlim(-1,1)
        plt.ylim(1,-1)
    
    plt.show()
    plt.close()