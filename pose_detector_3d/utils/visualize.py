#from __future__ import print_function, absolute_import, division

import matplotlib.pyplot as plt
import numpy as np
import cv2
import io

pos = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]

I   = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
J   = np.array([2,3,4,7,8,9,13,14,18,16,19,20,26,27,28])-1 # end points
LR  = np.array([1,1,1,0,0,0,0, 0,0, 0, 0, 0, 1, 1, 1], dtype=bool)

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_3D_pose(vals, show=False, azim=-90, elev=-140, name='test', lcolor="#3498db", rcolor="#e74c3c"): # blue, orange
    # ax = plt.subplot(111, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[pos.index(I[i]), j], vals[pos.index(J[i]), j]] ) for j in range(3)]
        # x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor, zdir='z')

    RADIUS = 1.3 # space around the subject
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

    if show:
        plt.show()
    else:
        return get_img_from_fig(fig)
        # plt.savefig('output/' + name + '.png')
    plt.close()

def show_3d_prediction(pred, gt, show=False, azim=-90, elev=-140, pcolor="#3498db", gtcolor="#e74c3c", name='test'): # blue, orange
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

    RADIUS = 1.3 # space around the subject
    xroot, yroot, zroot = gt[0,0], gt[0,1], gt[0,2]

    for ax in axx:
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