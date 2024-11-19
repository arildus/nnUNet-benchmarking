#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
from multiprocessing.pool import Pool
from typing import Tuple, Union

import numpy as np
import pandas as pd
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results, nnUNet_visualization
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets

color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None,
                     color_cycle: Tuple[str, ...] = color_cycle,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d greyscale or 2d RGB (color channel in last dimension!)

    Segmentation must be label map of same shape as image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255] (uint8)!!!
    """
    # create a copy of image
    image = np.copy(input_image)

    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            raise RuntimeError(f'if 3d image is given the last dimension must be the color channels (3 channels). '
                               f'Only 2D images are supported. Your image shape: {image.shape}')
    else:
        raise RuntimeError("unexpected image shape. only 2D images and 2D images with color channels (color in "
                           "last dimension) are supported")

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    # create output
    if mapping is None:
        uniques = np.sort(pd.unique(segmentation.ravel()))  # np.unique(segmentation)
        mapping = {i: c for c, i in enumerate(uniques)}

    for l in mapping.keys():
        image[segmentation == l] += overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))

    # rescale result to [0, 255]
    image = image / image.max() * 255
    return image.astype(np.uint8)


def select_slice_to_plot(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D

    selects the slice with the largest amount of fg (regardless of label)

    we give image so that we can easily replace this function if needed
    """
    fg_mask = segmentation != 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = int(np.argmax(fg_per_slice))
    return selected_slice


def select_slice_to_plot2(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D (or 1, x, y)

    selects the slice with the largest amount of fg (how much percent of each class are in each slice? pick slice
    with highest avg percent)

    we give image so that we can easily replace this function if needed
    """
    classes = [i for i in np.sort(pd.unique(segmentation.ravel())) if i != 0]
    fg_per_slice = np.zeros((image.shape[0], len(classes)))
    for i, c in enumerate(classes):
        fg_mask = segmentation == c
        fg_per_slice[:, i] = fg_mask.sum((1, 2))
        fg_per_slice[:, i] /= fg_per_slice.sum()
    fg_per_slice = fg_per_slice.mean(1)
    return int(np.argmax(fg_per_slice))


def plot_overlay(image_file: str, segmentation_file: str, image_reader_writer: BaseReaderWriter, output_file: str,
                 overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image, props = image_reader_writer.read_images((image_file, ))
    image = image[0]
    seg, props_seg = image_reader_writer.read_seg(segmentation_file)
    seg = seg[0]
    #print("raw image shape: ", image.shape)
    #print("ground truth data shape: ", seg.shape)

    assert image.shape == seg.shape, "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert image.ndim == 3, 'only 3D images/segs are supported'

    selected_slice = select_slice_to_plot2(image, seg)
    # print(image.shape, selected_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)


    image_dir = os.path.dirname(output_file)
    print(image_dir)
    maybe_mkdir_p(image_dir)

    #print(image_file)
    #print(segmentation_file)
    #print(output_file)
    
    plt.imsave(output_file, overlay)

    return selected_slice


def plot_overlay_prediction(image_file: str, segmentation_file: str, gt_segmentation_file: str, image_reader_writer: BaseReaderWriter, output_file: str, slice: int, overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image, props = image_reader_writer.read_images((image_file, ))
    image = image[0]
    seg, props_seg = image_reader_writer.read_seg(segmentation_file)
    seg = seg[0]
    gt_seg, props_seg = image_reader_writer.read_seg(gt_segmentation_file)
    gt_seg = gt_seg[0]
    #print("raw image shape: ", image.shape)
    #print("pred data shape: ", seg.shape)

    assert image.shape == seg.shape, "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert image.ndim == 3, 'only 3D images/segs are supported'

    selected_slice = select_slice_to_plot2(image, gt_seg)
    #print("raw image shape: ", image.shape)
    #print("gt_seg data shape: ", seg.shape)
    #print("image file path: ", image_file)
    #print("gt_seg path: ", gt_segmentation_file)

    # print(image.shape, selected_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    maybe_mkdir_p(os.path.dirname(output_file))

    #print(image_file)
    #print(segmentation_file)
    #print(output_file)

    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(case_file: str, output_file: str, overlay_intensity: float = 0.6, channel_idx=0):
    import matplotlib.pyplot as plt
    data = np.load(case_file)['data']
    seg = np.load(case_file)['seg'][0]

    assert channel_idx < (data.shape[0]), 'This dataset only supports channel index up to %d' % (data.shape[0] - 1)

    image = data[channel_idx]
    seg[seg < 0] = 0

    selected_slice = select_slice_to_plot2(image, seg)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def multiprocessing_plot_overlay(list_of_image_files, list_of_seg_files, image_reader_writer,
                                 list_of_output_files, overlay_intensity,
                                 num_processes=8):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_overlay, zip(
            list_of_image_files, list_of_seg_files, [image_reader_writer] * len(list_of_output_files),
            list_of_output_files, [overlay_intensity] * len(list_of_output_files)
        ))
        r.get()


def multiprocessing_plot_overlay_preprocessed(list_of_case_files, list_of_output_files, overlay_intensity,
                                              num_processes=8, channel_idx=0):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_overlay_preprocessed, zip(
            list_of_case_files, list_of_output_files, [overlay_intensity] * len(list_of_output_files),
                                                      [channel_idx] * len(list_of_output_files)
        ))
        r.get()

def multiprocessing_plot_overlay_prediction(list_of_case_files, list_of_pred_files, list_of_gt_segmentation_files, image_reader_writer, 
                                            list_of_output_files, overlay_intensity,
                                              num_processes=8):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_overlay_prediction, zip(
            list_of_case_files, list_of_pred_files, list_of_gt_segmentation_files, [image_reader_writer] * len(list_of_output_files), 
            list_of_output_files, [overlay_intensity] * len(list_of_output_files)
        ))
        r.get()


def generate_overlays_from_raw(dataset_name_or_id: Union[int, str], output_folder: str,
                               case: str, fold: int,
                               num_processes: int = 8, channel_idx: int = 0, overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(folder, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    if output_folder is None:
        output_folder = join(nnUNet_visualization, dataset_name)

    network_output_files_dict = {}
    network_prediction_files_dict = {}
    network_gt_files_dict = {}
    network_image_files_dict = {}
    identifiers = []

    image_files = []
    seg_files = []
    
    if fold is not None:

        results_dataset_folder_base = os.path.join(nnUNet_results, dataset_name) 
        trainer_result_folders = os.listdir(results_dataset_folder_base)

        for trainer in trainer_result_folders:
            fold_folder_base = os.path.join(results_dataset_folder_base, trainer, f'fold_{fold}')

            if not os.path.exists(fold_folder_base):
                print(f"Could not find folder '{fold_folder_base}'. Not able to visualize results.")
            else:
                debug_file_path = os.path.join(fold_folder_base, 'debug.json')
                #print(debug_file_path)

                if os.path.isfile(debug_file_path):
                    f = open(debug_file_path, "r")
                    data = json.load(f)
                    network_name = data['network']
                    #print('Network: ', network_name)
                else:
                    print(f"Could not find file '{debug_file_path}'. Network name could not be extracted. Trainer name used.")
                    network_name = str.join(trainer.split('_')[:-5])
                
                validation_folder_base = os.path.join(fold_folder_base, 'validation')

                if not os.path.exists(validation_folder_base):
                    print(f"Could not locate folder '{validation_folder_base}'. Make sure to run trainer with --val option to generate validation samples.")

                network_identifiers = []
                if case is not None:
                    if not os.path.isfile(join(validation_folder_base, case + '.nii.gz')):
                        #print(join(validation_folder_base, case + '.nii.gz'))
                        print(f"Could not locate case '{case}' for fold {fold} for network '{network_name}'.")
                    else:
                        network_identifiers = [case]
                else:
                    network_identifiers = [i[:-7] for i in subfiles(validation_folder_base, suffix='.gz', join=False)]

                network_output_files_dict[network_name] = [join(output_folder, i, i + '_' + network_name +'.png') for i in network_identifiers]
                network_prediction_files_dict[network_name] = [join(validation_folder_base, i + '.nii.gz') for i in network_identifiers]
                network_gt_files_dict[network_name] = [dataset[i]['label'] for i in network_identifiers]
                network_image_files_dict[network_name] = [dataset[i]['images'][channel_idx] for i in network_identifiers]

                identifiers.extend(network_identifiers)
        
        identifiers = list(set(identifiers))
        image_files = [dataset[i]['images'][channel_idx] for i in identifiers]
        seg_files = [dataset[i]['label'] for i in identifiers]

    elif case is None:
        image_files = [v['images'][channel_idx] for v in dataset.values()] # 'content/data/nnUNet_raw_data_base/Dataset137_BraTS2021/imagesTr/BraTS2021_00599_0000.nii.gz'
        seg_files = [v['label'] for v in dataset.values()]                 # 'content/data/nnUNet_raw_data_base/Dataset137_BraTS2021/labelsTr/BraTS2021_00599.nii.gz'
    else:
        if case not in dataset.keys():
            #print(join(preprocessed_folder, case + '.npz'))
            print(f"Could not locate case '{case}' in raw_data_base folder.")
        else:
            image_files = [dataset[case]['images'][channel_idx]]
            seg_files = [dataset[case]['label']]
    


    #print("image_files: ", image_files)
    #print("seg_files: ", seg_files)
    #print("identifiers: ", identifiers)

    assert all([isfile(i) for i in image_files])
    assert all([isfile(i) for i in seg_files])

    gt_output_files = [join(output_folder, i, i + '_gt.png') for i in identifiers]
    #print("gt_ouput_files: ", gt_output_files)

    image_reader_writer = determine_reader_writer_from_dataset_json(dataset_json, image_files[0])()
    multiprocessing_plot_overlay(image_files, seg_files, image_reader_writer, gt_output_files, overlay_intensity, num_processes)
    

    if fold is not None:
        for network, output_files in network_output_files_dict.items():
            print("Network: ", network)
            #print(output_files)
            list_of_prediction_files = network_prediction_files_dict[network]
            list_of_gt_segmentation_files = network_gt_files_dict[network]
            list_of_image_files = network_image_files_dict[network]
            #print(list_of_prediction_files)
            multiprocessing_plot_overlay_prediction(list_of_image_files, list_of_prediction_files, list_of_gt_segmentation_files, image_reader_writer, 
                                                    output_files, overlay_intensity=overlay_intensity, 
                                                    num_processes=num_processes)


def generate_overlays_from_preprocessed(dataset_name_or_id: Union[int, str], output_folder: str,
                                        num_processes: int = 8, channel_idx: int = 0,
                                        configuration: str = None,
                                        plans_identifier: str = 'nnUNetPlans',
                                        overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnUNet_preprocessed, dataset_name)

    if output_folder is None:
        output_folder = join(nnUNet_visualization, dataset_name)

    if not isdir(folder): raise RuntimeError("run preprocessing for that task first")

    plans = load_json(join(folder, plans_identifier + '.json'))
    if configuration is None:
        if '3d_fullres' in plans['configurations'].keys():
            configuration = '3d_fullres'
        else:
            configuration = '2d'
    data_identifier = plans['configurations'][configuration]["data_identifier"]
    preprocessed_folder = join(folder, data_identifier)

    if not isdir(preprocessed_folder):
        raise RuntimeError(f"Preprocessed data folder for configuration {configuration} of plans identifier "
                           f"{plans_identifier} ({dataset_name}) does not exist. Run preprocessing for this "
                           f"configuration first!")

    identifiers = [i[:-4] for i in subfiles(preprocessed_folder, suffix='.npz', join=False)]

    output_files = [join(output_folder, i + '.png') for i in identifiers]
    image_files = [join(preprocessed_folder, i + ".npz") for i in identifiers]

    maybe_mkdir_p(output_folder)
    multiprocessing_plot_overlay_preprocessed(image_files, output_files, overlay_intensity=overlay_intensity,
                                              num_processes=num_processes, channel_idx=channel_idx)


def entry_point_generate_overlay():
    import argparse
    parser = argparse.ArgumentParser("Plots png overlays of the slice with the most foreground. Note that this "
                                     "disregards spacing information!")
    parser.add_argument('-d', type=str, help="Dataset name or id", required=True)
    parser.add_argument('-o', type=str, default=None, help="output folder", required=False)
    parser.add_argument('-case', type=str, default=None, help="Case to overlay", required=False)
    parser.add_argument('-val_fold', type=int, default=None, help="If you want to overlay predictions of a certain fold.  Should be an int between 0 and 4.", required=False)
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f"number of processes used. Default: {default_num_processes}")
    parser.add_argument('-channel_idx', type=int, default=0, required=False,
                        help="channel index used (0 = _0000). Default: 0")
    parser.add_argument('--use_raw', action='store_true', required=False, help="if set then we use raw data. else "
                                                                               "we use preprocessed")
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='plans identifier. Only used if --use_raw is not set! Default: nnUNetPlans')
    parser.add_argument('-c', type=str, required=False, default=None,
                        help='configuration name. Only used if --use_raw is not set! Default: None = '
                             '3d_fullres if available, else 2d')
    parser.add_argument('-overlay_intensity', type=float, required=False, default=0.6,
                        help='overlay intensity. Higher = brighter/less transparent')


    args = parser.parse_args()

    if args.use_raw:
        generate_overlays_from_raw(args.d, args.o, args.case, args.val_fold, args.np, args.channel_idx,
                                   overlay_intensity=args.overlay_intensity)
    else:
        generate_overlays_from_preprocessed(args.d, args.o, args.np, args.channel_idx, args.c, args.p,
                                            overlay_intensity=args.overlay_intensity)


if __name__ == '__main__':
    entry_point_generate_overlay()