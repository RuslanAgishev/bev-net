import os
import matplotlib.pyplot as plt
import numpy as np
import imageio


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def visualize_points_prediction(points,
                                cat_pred,
                                disp_pred,
                                voxel_size,
                                ax,
                                img_save_dir='logs',
                                fname='result.png'):
    # --- Visualization ---
    # Draw the LIDAR and quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
    cat_names = {0: 'bg', 1: 'vehicle', 2: 'ped', 3: 'bike', 4: 'other'}

    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]

    ax[0].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1)
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    ax[0].title.set_text('LIDAR data')

    for k in [0, 2]: #range(len(color_map)):
        # ------------------------ Prediction ------------------------
        # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
        mask_pred = cat_pred == (k + 1)
        field_pred = disp_pred[-1]  # Show last prediction, ie., the 20-th frame

        # For cells with very small movements, we threshold them to be static
        field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_pred_norm <= 0.4
        field_pred[thd_mask, :] = 0

        # Plot quiver. We only show non-empty vectors. Plot each category.
        idx_x = np.arange(field_pred.shape[0])
        idx_y = np.arange(field_pred.shape[1])
        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
        qk = [None] * len(color_map)  # for quiver key

        # We use the same indices as the ground-truth, since we are currently focused on the foreground
        X_pred = idx_x[mask_pred]
        Y_pred = idx_y[mask_pred]
        U_pred = field_pred[:, :, 0][mask_pred] / voxel_size[0]
        V_pred = field_pred[:, :, 1][mask_pred] / voxel_size[1]

        ax[1].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])
        ax[1].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
        ax[1].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
        ax[1].set_aspect('equal')
        ax[1].title.set_text('Prediction')
        ax[1].axis('off')
    # img_save_dir = check_folder(img_save_dir)
    # plt.savefig(os.path.join(img_save_dir, fname))


def visualize_points(points):
    # --- Visualization ---
    # Draw the LIDAR plot
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]

    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1)
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.axis('off')
    plt.title('LIDAR data')

def visualize_prediction(cat_pred,
                         disp_pred,
                         voxel_size,
                         movement_pred_th=0.4,
                         file_savepath=None):
    # --- Visualization ---
    # Draw quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
    cat_names = {0: 'bg', 1: 'vehicle', 2: 'ped', 3: 'bike', 4: 'other'}

    border_meter = 4
    border_pixel = border_meter * 4

    for k in range(len(color_map)):
        # ------------------------ Prediction ------------------------
        # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
        mask_pred = cat_pred == (k + 1)
        field_pred = disp_pred[-1]

        # For cells with very small movements, we threshold them to be static
        field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_pred_norm <= 0.4
        field_pred[thd_mask, :] = 0

        # Plot quiver. We only show non-empty vectors. Plot each category.
        idx_x = np.arange(field_pred.shape[0])
        idx_y = np.arange(field_pred.shape[1])
        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')

        # We use the same indices as the ground-truth, since we are currently focused on the foreground
        X_pred = idx_x[mask_pred] * voxel_size[0]
        Y_pred = idx_y[mask_pred] * voxel_size[1]
        U_pred = field_pred[:, :, 0][mask_pred]  # / voxel_size[0]
        V_pred = field_pred[:, :, 1][mask_pred]  # / voxel_size[1]

        plt.quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])

    plt.xlim(border_meter, field_pred.shape[0] * voxel_size[0] - border_meter)
    plt.ylim(border_meter, field_pred.shape[1] * voxel_size[1] - border_meter)
    plt.title('Prediction: local map')
    plt.plot(field_pred.shape[0]*voxel_size[0]/2.,
             field_pred.shape[1]*voxel_size[1]/2.,
             '*', markersize=10, color='g', label='Ego position')
    plt.grid()
    plt.legend()
    # plt.axis('off')
    if file_savepath is not None:
        plt.savefig(file_savepath)

def gen_scene_prediction_video(images_dir, output_dir, out_format='gif'):
    images = [im for im in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, im))
              and im.endswith('.png')]
    num_images = len(images)
    if out_format == 'gif':
        save_gif_path = os.path.join(output_dir, 'result.gif')
        with imageio.get_writer(save_gif_path, mode='I', fps=20) as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + '.png')
                image = imageio.imread(image_file)
                writer.append_data(image)
                print("Appending image {}".format(i))
    else:
        save_mp4_path = os.path.join(output_dir, 'result.mp4')
        with imageio.get_writer(save_mp4_path, fps=15, quality=10, pixelformat='yuvj444p') as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + '.png')
                image = imageio.imread(image_file)
                writer.append_data(image)
                print("Appending image {}".format(i))
