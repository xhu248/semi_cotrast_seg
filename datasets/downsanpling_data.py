from utilities.file_and_folder_operations import subfiles
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import math

def reshape_array(numpy_array, axis=1):
    shape = numpy_array.shape[1]
    if axis == 1:
        slice_img = numpy_array[:, 0, :, :].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[1]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, k, :, :].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 2:
        slice_img = numpy_array[:, :, 0, :].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[2]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, k, :].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 3:
        slice_img = numpy_array[:, :, :, 0].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[3]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, :, k].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img


def downsampling_image(data_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    npy_files = subfiles(data_dir, suffix=".npy", join=False)
    for file in npy_files:
        np_path = os.path.join(data_dir, file)
        save_path = os.path.join(output_dir, file.split('.')[0] + '.npy')

        if not os.path.exists(save_path):
            numpy_array = reshape_array(np.load(np_path), axis=3)
            shape = numpy_array.shape[3]
            num_of_pooling = math.ceil(math.log(shape, 2)) - 4

            ################ test num_of_pooling ###############
            num_of_pooling = num_of_pooling - 1

            slice_data = torch.from_numpy(numpy_array).to(device)

            for k in range(num_of_pooling):
                # pooling_data = F.max_pool2d(slice_data, kernel_size=2, stride=2)
                pooling_data = F.interpolate(slice_data, scale_factor=1/2, mode='bilinear')
                slice_data = pooling_data

            pooling_array = slice_data.cpu().numpy()
            np.save(os.path.join(output_dir, file.split('.')[0] + '.npy'), pooling_array)
            print(file)

    # else:
    #    print("scaled image has already been created")

    # data_path = os.path.join(project_dir, "data/Task01_BrainTumour/preprocessed")
    # image_dir = os.path.join(c.data_dir)


""""
file_num = len(npy_files)
for i in range(1, 50):
    np_path = os.path.join(data_path, npy_files[i])
    numpy_array = np.load(np_path)
    slice_data = reshape_array(numpy_array)
    slice_img = np.concatenate((slice_img, slice_data))

print(np.shape(slice_img))




pooling_1_data = F.max_pool2d(batch_data, kernel_size=2, stride=2)
pooling_2_data = F.max_pool2d(pooling_1_data, kernel_size=2, stride=2)

batch_image = batch_data[150]
plt.figure(1)
plt.subplot(3, 2, 1)
plt.imshow(batch_image[0], cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(batch_image[1], cmap='gray')


pooling_image = pooling_1_data[150]
plt.figure(1)
plt.subplot(3, 2, 3)
plt.imshow(pooling_image[0], cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(pooling_image[1], cmap='gray')

pooling_image = pooling_2_data[150]
plt.figure(1)
plt.subplot(3, 2, 5)
plt.imshow(pooling_image[0], cmap='gray')
plt.subplot(3, 2, 6)
plt.imshow(pooling_image[1], cmap='gray')

plt.show()

"""