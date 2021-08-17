import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


class misc():

    @staticmethod
    def is_notebook():
        notebooks = ("<class 'google.colab._shell.Shell'>",
                     "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>")
        try:
            shell_type = str(type(get_ipython()))
            if shell_type in notebooks:
                return True
            else:
                return False
        except:
            return False

    @staticmethod
    def set_torch_device():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return (device)

    @staticmethod
    def sample_plot(figsize,
                    dpi,
                    ck,
                    lm,
                    cbaru,
                    cbard,
                    file=None,
                    colormap='inferno',
                    s=50,
                    label=None,
                    label_no='',
                    show=True):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        im = ax.imshow(ck, cmap=colormap, vmin=cbard, vmax=cbaru)
        ax.set_ylabel(r'$\longleftarrow$ $t_j$')
        ax.set_xlabel(r'$t_i$ $\longrightarrow$')
        ax.xaxis.set_label_position('top')
        ax.set_xticks(np.arange(0, ck.shape[1], int(ck.shape[1] / 10)))
        ax.set_yticks(np.arange(0, ck.shape[1], int(ck.shape[1] / 10)))
        ax.xaxis.tick_top()
        ax.scatter(x=lm[0], y=lm[1], s=s, c='w')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if label != None:
            clabel = r"$S_{0}\left(t_i, t_j\right)$".format(label)
        else:
            clabel = r"$S\left(t_i, t_j\right)$"
        cbar.set_label(clabel)
        if file != None:
            fig.savefig("{0}_{1}.pdf".format(file, str(label_no)),
                        format='pdf',
                        bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def single_plot(ck,
                    ck_error,
                    lm,
                    figsize=(12, 6),
                    dpi=80,
                    file=None,
                    grid=True,
                    linewidth=0.3,
                    s=50,
                    label='',
                    label_no='',
                    show=True):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.grid(grid, linestyle=":")
        plt.plot(range(2,
                       len(ck) + 2),
                 ck,
                 c='k',
                 alpha=0.7,
                 linewidth=linewidth,
                 label=r'$S_{0}(\tau)$'.format(label))
        plt.plot(range(2, len(ck) + 2), ck_error, c='r', label=r'$\sigma_S$', linewidth=linewidth)
        plt.scatter(lm + 1, ck[lm - 1], zorder=10, label=r'$l_m = {0}$'.format(lm), s=s, alpha=0.7)
        plt.ylabel(r'$S_{0}(\tau)$'.format(label))
        plt.xlabel(r'$\tau$')
        plt.legend()
        if file != None:
            plt.savefig("{0}_{1}.pdf".format(file, str(label_no)),
                        format='pdf',
                        bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def write(_):
        pass
