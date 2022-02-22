import numpy as np
import os
import pickle
import cv2

def load_dataset(path):
    with open(path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    label = batch[b'labels']
    data = batch[b'data']

    return label, data


def save_png(label, data, path, class_id_name_dict):
    total_counts = len(label)
    for idx, val in enumerate(label):
        sample_image = data[idx,:].reshape((3, 32, 32)).astype('uint8').transpose(1, 2, 0)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
        sample_name = f'{idx:05}.png'
        cv2.imwrite(f'{path}/{class_id_name_dict[val]}/{sample_name}', sample_image)
        
        count = idx + 1
        if count % 1000 == 0:
            print(f'{count}/{total_counts}')

def preprocess_cifar10():
    train_base = '/home/jovyan/cifar10-practice/data/raw/cifar-10-batches-py/data_batch_'
    test_path = '/home/jovyan/cifar10-practice/data/raw/cifar-10-batches-py/test_batch'
    png_output_dir = '/home/jovyan/cifar10-practice/data/processed/cifar10_original_png'
    png_train_dir = f'{png_output_dir}/train'
    png_test_dir = f'{png_output_dir}/test'

    class_name_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_id_name_dict = dict()
    for idx, name in enumerate(class_name_list):
        class_id_name_dict.update({idx:name})
        os.makedirs(f'{png_train_dir}/{name}', exist_ok=True)
        os.makedirs(f'{png_test_dir}/{name}', exist_ok=True)

    print(f'classify data : {class_id_name_dict}')
    
    
    test_label, test_data = load_dataset(test_path)
    print(f'count test data by class: {np.unique(test_label, return_counts = True)}')

    train_label = np.empty([0])
    train_data = np.empty([0, 32 * 32 * 3])

    for i in range(1, 6):
        batch_path = f'{train_base}{i}'
        batch_label, batch_data = load_dataset(batch_path)

        train_label = np.concatenate((train_label, batch_label), axis=0)
        train_data = np.concatenate((train_data, batch_data), axis=0)

    print(f'count train data by class: {np.unique(train_label, return_counts = True)}')
    save_png(test_label, test_data, png_test_dir, class_id_name_dict)
    save_png(train_label, train_data, png_train_dir, class_id_name_dict)
    


if __name__ == "__main__":
    preprocess_cifar10()