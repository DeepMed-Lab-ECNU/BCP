import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from skimage import measure

"""
  2D Image im_show in one graph 
"""

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
marker_types = ['.', 'v', 's', '*', 'p', 'H', 'X', '1', '8', ]


def show_graphs(imgs, titles=None, show=True, filename=None, figsize=(5, 5), bbox=[], colors=[], show_type='gray'):
    """  Show images in a grid manner. it will automatically get a almost squared grid to show these images.

    :param imgs: input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param titles: [str, ...], the title for every image
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    col = np.ceil(np.sqrt(len(imgs)))
    show_graph_with_col(imgs, max_cols=col, titles=titles, show=show, filename=filename, figsize=figsize, bbox=bbox,
                        colors=colors, show_type=show_type)


def show_graph_with_col(imgs, max_cols, titles=None, show=True, filename=None, figsize=(5, 5),
                        bbox=[], colors=[], show_type='gray'):
    """ Show images in a grid manner.

    :param imgs: assume shape with [N, C, D, H, W], [N, C, H, W], [C, H, W], [N, H, W], [H, W]
             input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param max_cols: int, max column of grid.
    :param titles: [str, ...], the title for every image
    :param show:  True or False, show or save image
    :param filename: str, if save image, specify the path
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    """
    Check size and type
    """
    if len(imgs) == 0:
        return

    length = len(imgs)
    if length < max_cols:
        max_cols = length

    img = imgs[0]
    if isinstance(img, np.ndarray):
        shape = img.shape
    elif isinstance(img, torch.Tensor):
        shape = img.size()
    else:
        raise Exception("Unknown type of imgs : {}".format(type(imgs)))
    assert 2 <= len(shape) <= 5, 'Error shape : {}'.format(shape)

    """
    Plot graph
    """
    fig = plt.figure(figsize=figsize)
    max_line = np.ceil(length / max_cols)
    for i in range(1, length + 1):
        ax = fig.add_subplot(max_line, max_cols, i)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if titles is not None:
            ax.set_title(titles[i - 1])

        img = imgs[i - 1]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = img.copy()
        img[img == -1] = 0
        color = False
        shape = img.shape
        if len(shape) == 5:
            # maybe colored image
            if shape[1] == 3:
                color = True
                img = img[0, :, 0, :, :]
            else:
                img = img[0, 0, 0, :, :]
        if len(shape) == 4:
            if shape[1] == 3:
                color = True
                img = img[0]
            else:
                img = img[0, 0]
        elif len(shape) == 3:
            if shape[0] == 3:
                color = True
            else:
                img = img[0]

        if color:
            # normalized image
            if img.min() != 0:
                img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                        np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255).astype(np.int32)
            img = img.transpose((1, 2, 0)).astype(np.int32)
            ax.imshow(img)
        else:
            if show_type == 'gray' or show_type == 'hot' or show_type is None:
                ax.imshow(img, cmap=show_type)
            elif show_type[:4] == 'hot_':
                vmin = int(show_type[4])
                vmax = int(show_type[5])
                ax.imshow(img, cmap=show_type[:3], vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, cmap=show_type)

        for i, box in enumerate(bbox):
            (min_x, max_x), (min_y, max_y) = box
            if len(colors) == len(bbox):
                color = colors[i]
            else:
                color = COLORS[i % len(COLORS)]
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(rect)

    # plt.subplots_adjust(wspace=0, hspace=0)
    if filename is not None:
        plt.savefig(filename)  # , bbox_inches='tight'
    if show:
        plt.show()
    plt.close()


def torch_im_show_3d(input, label, logits, name='img', should_sigmoid=True):
    """ Show 3d images of the (input, label, logits). If should_sigmoid is true,
    it will perform sigmoid or softmax first to get final output.
    :param input: torch.Tensor with shape [N, C, D, H, W]
    :param label: torch.Tensor with shape [N, 1, D, H, W]
    :param logits: torch.Tensor with shape [N, C, D, H, W] if should_sigmoid is true,
                   else [N, D, H, W]
    :param name: the data name of input, normally the filename should be passed in.
    :param should_sigmoid: whether to perform sigmoid or softmax
    """
    depth = label.size()[2]
    for i in range(depth):
        if not should_sigmoid:
            torch_im_show(input[:, :, i, :, :], label[:, :, i, :, :], logits[:, i, :, :], name, should_sigmoid)
        else:
            torch_im_show(input[:, :, i, :, :], label[:, :, i, :, :], logits[:, :, i, :, :], name, should_sigmoid)


def torch_im_show(input, label, logits, name='img', should_sigmoid=True):
    """ Only show one image of the (input, label, logits). If should_sigmoid is true,
    it will perform sigmoid or softmax first to get final output.
    :param input: torch.Tensor with shape [N, C, H, W]
    :param label: torch.Tensor with shape [N, 1, H, W]
    :param logits: torch.Tensor with shape [N, C, H, W] if should_sigmoid is true,
                   else [N, H, W]
    :param name: the data name of input, normally the filename should be passed in.
    :param should_sigmoid: whether to perform sigmoid or softmax
    """
    if isinstance(logits, tuple) or isinstance(logits, list):
        logits = logits[-1]

    # get prediction
    if should_sigmoid:
        pred = get_prediction(logits)
    else:
        pred = logits
    # to numpy
    input = input.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    # reduce the batch and channel dim. Only input H and W dim.
    if input.shape[1] == 3:
        input = input[0]
    else:
        input = input[0, 0]
    show_graphs([input, pred[0], label[0, 0]], [name, 'pred', 'mask'])


def show_landmark_2d(mask, landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l = mlines.Line2D((landmarks[0, 0], landmarks[1, 0]), (landmarks[0, 1], landmarks[1, 1]))
    ax.add_line(l)
    plt.imshow(mask, cmap='gray')
    plt.show()


def show_landmark_3d(landmarks):
    x_start = landmarks[::2, 0]
    x_end = landmarks[1::2, 0]
    y_start = landmarks[::2, 1]
    y_end = landmarks[1::2, 1]
    z_start = landmarks[::2, 2]
    z_end = landmarks[1::2, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x_start)):
        ax.plot([x_start[i], x_end[i]], [y_start[i], y_end[i]], zs=[z_start[i], z_end[i]])
    plt.show()


def show_3d(mask):
    verts, faces, _, _ = measure.marching_cubes_lewiner(mask, 0, spacing=(0.1, 0.1, 0.1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
    plt.show()


def show_segmentation(mask):
    z, y, x = np.where(mask)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def label_overlay_torch(img, label, save_path=None, color=(1.0, 0., 0.), fill=True, show_type='gray'):
    H, W = img.size()[2:]
    label = F.interpolate(label.type(torch.float32), (H, W), mode='nearest')
    img = (img.detach().cpu().numpy()[0, 0] * 255).astype(np.int32)
    label = label.detach().cpu().numpy()[0, 0].astype(np.uint8)
    label_overlay(img, label, save_path, color, fill, show_type)


def label_overlay(img, label, save_path=None, color=(1.0, 0., 0.), fill=False, show_type='gray'):
    """
    :param img:   (ndarray, 3xhxw, or hxw)
    :param label: (ndarray, hxw)
    :return:
    """
    if len(img.shape) == 3:
        img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255)
        img = img.transpose(1, 2, 0).astype(np.int32)
    label = (label > 0).astype(np.uint8)

    fig = plt.figure(frameon=False, figsize=(5, 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img, cmap=show_type)

    import cv2
    from matplotlib.patches import Polygon

    contour, hier = cv2.findContours(label.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contour:
        ax.add_patch(
            Polygon(
                c.reshape((-1, 2)),
                fill=fill, facecolor=color, edgecolor='r', linewidth=2.0, alpha=0.5
            )
        )
    if save_path is None:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        fig.savefig(save_path)
    plt.close('all')


class ImageOverlay(object):
    def __init__(self, img, cmap='gray'):
        assert isinstance(img, np.ndarray), len(img.shape) in [2, 3]

        if len(img.shape) == 3:
            img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                    np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255)
            img = img.transpose(1, 2, 0).astype(np.int32)

        fig = plt.figure(frameon=False, figsize=(5, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(img, cmap=cmap)

        self.fig = fig
        self.ax = ax

    def overlay(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        assert isinstance(mask, np.ndarray), len(mask.shape) == 2

        import cv2
        from matplotlib.patches import Polygon

        mask = (mask > 0).astype(np.uint8)
        # _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contour:
            self.ax.add_patch(
                Polygon(
                    c.reshape((-1, 2)),
                    fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
                )
            )
        return self

    def overlay_hole(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        import cv2
        from matplotlib.path import Path

        mask = self.to_numpy(mask)
        mask = (mask > 0).astype(np.uint8)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        path_points = []
        path_move = []
        for c in contour:
            c = c.reshape(-1, 2)
            for i, p in enumerate(c):
                path_points.append(p)
                if i == 0:
                    path_move.append(Path.MOVETO)
                elif i == len(c) - 1:
                    path_move.append(Path.CLOSEPOLY)
                else:
                    path_move.append(Path.LINETO)
        from matplotlib.patches import PathPatch
        patch = PathPatch(Path(path_points, path_move), fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        self.ax.add_patch(patch)
        return self

    def show(self):
        plt.show()
        return self

    def save(self, save_path):
        self.fig.savefig(save_path)
        return self


def show_scatter(tsne_X, label, label_name=None, marker_size=2, marker_type='o', imgs=None, ax=None):
    # imgs : 3 x H x W
    # label : N
    # tsne_X : N x 2
    if ax is None:
        ax = plt.gca()
    label = np.array(label)

    if imgs is None:
        # colors = np.array(sns.color_palette("husl", label.max()+1))
        # plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=colors[label], s=marker_size, label=label, marker=marker_type)
        label[label == 5] = 7  # 5太黄了
        plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=plt.cm.Set1(label), s=marker_size, label=label, marker=marker_type)
    else:
        imgs = imgs.swapaxes(1, 2).swapaxes(2, 3)  # to channel last
        for i, (img, (x0, y0)) in enumerate(zip(imgs, tsne_X)):
            img = ((img * np.array([.229, .224, .225]).reshape(1, 1, 3) + np.array([.485, .456, .406]).reshape(1, 1, 3)) * 255).astype(np.int32)
            img = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)

    # plot legend
    each_labels = np.unique(label)
    legend_elements = []
    for l in each_labels:
        if label_name is not None:
            L = label_name[l]
        else:
            L = str(l)
        legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', label=L, markerfacecolor=plt.cm.Set1(l), markersize=5))
        plt.legend(handles=legend_elements)


class tSNE():
    @staticmethod
    def get_tsne_result(X, metric='euclidean', perplexity=30):
        """  Get 2D t-SNE result with sklearn

        :param X: feature with size of N x C
        :param metric: 'cosine', 'euclidean', and so on.
        :param perplexity:  the preserved local structure size
        """
        from sklearn.manifold.t_sne import TSNE
        tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity)
        tsne_X = tsne.fit_transform(X)
        tsne_X = (tsne_X - tsne_X.min()) / (tsne_X.max() - tsne_X.min())
        return tsne_X

    @staticmethod
    def plot_tsne(tsne_X, labels, domain_labels=None, imgs=None, save_name=None, figsize=(10, 10), marker_size=20):
        """ plot t-SNE results. All parameters are numpy format.

        Args:
            tsne_X: N x 2
            labels: N
            domain_labels: N
            imgs: N x 3 x H x W
            save_name: str
            figsize: tuple of figure size
            marker_size: size of markers
        """
        plt.figure(figsize=figsize)
        if domain_labels is not None:
            # plot each domain with different shape of markers
            domain_num = np.unique(domain_labels).max()
            for i in range(domain_num + 1):
                idx = domain_labels == i
                x_tmp = imgs[idx] if imgs is not None else None
                show_scatter(tsne_X[idx], labels[idx], None, marker_size=marker_size, marker_type=marker_types[i], imgs=x_tmp)
        else:
            # plot simple clusters of classes with different colors
            show_scatter(tsne_X, labels, None, marker_size=marker_size, marker_type=marker_types[0], imgs=imgs)

        if save_name is not None:
            plt.savefig(save_name)
        else:
            plt.show()


def bar_plot(data, x_aixs_names=None, bar_names=None, width_of_all_col=0.7, offset_between_bars=0.0,
             text_size='large', x_label=None, y_label=None, title=None,
             ylim=None, figsize=(10, 6), save_name=None, x_axis_name_rotate=None):
    """
    Args:
        data: shape of (Columns, DataNum)  (e.g., 3 methods x outputs of each method)
        x_aixs_names: name of each data, same shape of DataNum
        bar_names:    name of each column  (e.g., method names)
        width_of_all_col: total width of all types of data ( all_width = bar_width * columns + offset * (columns-1))
        offset_between_bars: offset between data
        text_size : size of text over the bar, choices : [ll, small, medium, large, x-large, xx-large, larger]
        ylim: min and max of y-axis
        x_label: name of all x-axis
        y_label: name of all y-axis
        figsize: figure size
        save_name ; provide file name to save

    Examples:
        >>> domains = [[53.7, 51.8, 53.9, 54.8, 55.0, 55.1],
        >>>            [55.95, 57.19, 57.65, 57.94, 57.84, 57.64],
        >>>            [55.95, 57.19, 57.76, 57.87, 57.94, 57.67]
        >>>        ]
        >>> names = [str(i) for i in ['baseline', 1, 2, 4, 8, 16]]
        >>> labels = ['AGG', 'MLDG', 'MLDG+SIB']
        >>> ylim = [50, 60]
        >>> bar_plot(domains, names, labels, ylim=ylim)

    """
    colors = ['#5A9BD5', '#FF9966', '#1B9E78', '#ff585d']  # TODO : add more colors
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    num_of_cols, DataNum = data.shape

    if bar_names is None:
        bar_names = [None] * num_of_cols

    offset_between_col = offset_between_bars
    width_of_one_col = width_of_all_col / num_of_cols
    # start_x = start_x  #(width_of_one_col / 2 * (num_of_cols-1)) #(width_of_all_col - width_of_one_col) / num_of_cols
    start_x = np.arange(DataNum)
    plt.figure(figsize=figsize)
    for i, (l, c, label) in enumerate(zip(data, colors, bar_names)):
        # plot one bar
        h = plt.bar(start_x + width_of_one_col * i, l,
                    width=width_of_one_col - offset_between_col, color=c, label=label, linewidth=2.0)  # s-:方形

        # plot acc text over the bar
        for j, rect in enumerate(h):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{:.1f}'.format(data[i][j]), ha='center', va='bottom', size=text_size)

    # 'll, small, medium, large, x-large, xx-large, larger'
    plt.ylim(ylim)
    if x_aixs_names is None:
        x_aixs_names = np.arange(DataNum)
    plt.xticks(np.arange(DataNum) + width_of_all_col / 2 - width_of_one_col / 2, x_aixs_names, size=text_size, rotation=x_axis_name_rotate)
    plt.yticks(size=text_size)
    plt.xlabel(x_label, size=text_size)  # 横坐标名字
    plt.ylabel(y_label, size=text_size)  # 纵坐标名字
    plt.title(title, size=text_size)
    if bar_names[0] is not None:
        plt.legend()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
