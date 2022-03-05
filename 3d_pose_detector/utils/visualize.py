from __future__ import print_function, absolute_import, division

import matplotlib.pyplot as plt
import numpy as np


def show3Dpose(vals, name, show=False, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
    ax = plt.subplot(111, projection='3d')

    pos = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]

    I   = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
    J   = np.array([2,3,4,7,8,9,13,14,18,16,19,20,26,27,28])-1 # end points
    LR  = np.array([1,1,1,0,0,0,0, 0,0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[pos.index(I[i]), j], vals[pos.index(J[i]), j]] ) for j in range(3)]
        # x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor, zdir='y')

    RADIUS = 1.7 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.view_init(elev=-140., azim=-90)
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    if show:
        plt.show()
    else:
        plt.savefig('output/' + name + '.png')
    plt.close()
    #plt.show()