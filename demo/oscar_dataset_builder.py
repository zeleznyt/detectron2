# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import ast
import base64
import json
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import shutil
import random

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8
    cfg.freeze()
    return cfg


# def get_parser():
#     parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
#     parser.add_argument(
#         "--config-file",
#         default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
#     parser.add_argument("--video-input", help="Path to video file.")
#     parser.add_argument(
#         "--input",
#         nargs="+",
#         help="A list of space separated input images; "
#         "or a single glob pattern such as 'directory/*.jpg'",
#     )
#     parser.add_argument(
#         "--output",
#         help="A file or directory to save output visualizations. "
#         "If not given, will show output in an OpenCV window.",
#     )
#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.5,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--min-detected",
#         default=10,
#         help="Minimum number of instances for image to generate files",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )
#     parser.add_argument(
#         "--input-path",
#         default="datasets/coco_images",
#         help="path to file with input images",
#     )
#     parser.add_argument(
#         "--id-dictionary",
#         default=""
#     )
#     parser.add_argument(
#         "--output-features",
#         help="A directory to save output features. "
#     )
#     parser.add_argument(
#         "--coco-classnames",
#         default="/home/tomas/fav/dp/ms_coco_classnames.txt",
#         help="File with coco classes names"
#     )
#     return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def prepare_files(directory, prefix=''):
    with open(os.path.join(directory, '{}.feature.tsv'.format(prefix)), 'w') as f:
        pass
    with open(os.path.join(directory, '{}.label.tsv'.format(prefix)), 'w') as f:
        pass
    with open(os.path.join(directory, '{}.feature.lineidx'.format(prefix)), 'w') as f:
        pass
    with open(os.path.join(directory, '{}.label.lineidx'.format(prefix)), 'w') as f:
        pass
    with open(os.path.join(directory, '{}.yaml'.format(prefix)), 'w') as f:
        f.write('img: img.tsv\nhw: hw.tsv\nlabel: {}label.tsv\nfeature: {}.feature.tsv'.format(prefix, prefix))
    with open(os.path.join(directory, '{}.outliers.txt'.format(prefix)), 'w') as f:
        pass


def save_features(directory, idx, predictions, len_feature, len_label, coco_classnames, prefix=''):

    n_instances = len(predictions["instances"])
    if n_instances < 1:
        with open(os.path.join(directory, '{}.outliers.txt'.format(prefix)), 'a') as f:
            f.write('{}\n'.format(idx))
        return len_feature, len_label

    with open(os.path.join(directory, '{}.feature.lineidx'.format(prefix)), 'a') as f:
        f.write('{}\n'.format(len_feature))
    with open(os.path.join(directory, '{}.label.lineidx'.format(prefix)), 'a') as f:
        f.write('{}\n'.format(len_label))
        
    features_array = predictions["features"].cpu().detach().numpy()

    img_height = predictions["instances"].image_size[0]
    img_width = predictions["instances"].image_size[1]
    pred_boxes = np.reshape(list(predictions["instances"]._fields["pred_boxes"].tensor.cpu().numpy()), (n_instances, 4))
    x1 = np.reshape(pred_boxes[:, 0] / img_width, (n_instances, 1))
    y1 = np.reshape(pred_boxes[:, 1] / img_height, (n_instances, 1))
    x2 = np.reshape(pred_boxes[:, 2] / img_width, (n_instances, 1))
    y2 = np.reshape(pred_boxes[:, 3] / img_height, (n_instances, 1))
    x_len = x2-x1
    y_len = y2-y1

    features_array = np.concatenate((features_array, x1, y1, x2, y2, x_len, y_len), axis=1)
    # np.append(features_array, (np.reshape(pred_boxes[:, 0] / img_width, (8, 1))), axis=1)

    sb = base64.b64encode(features_array)
    # To decode (from Oscar):
    # np.frombuffer(base64.b64decode(sb), np.float32).reshape((8, -1))

    s = sb.decode("utf-8")
    features_info = {"num_boxes": features_array.shape[0], "features": s}

    features_file = os.path.join(directory, '{}.feature.tsv'.format(prefix))
    with open(features_file, 'a') as f:
        text = '{}\t{}\n'.format(idx, json.dumps(features_info))
        len_feature = len_feature + len(text)
        f.write(text)

    with open(coco_classnames) as f:
        content = f.read()
    coco_classnames = ast.literal_eval(content)

    labels_info = []
    for i in range(n_instances):
        labels_info.append({"class": coco_classnames[predictions["instances"]._fields["pred_classes"].cpu().numpy()[i]+1],
                          "rect": [float(i) for i in list(predictions["instances"]._fields["pred_boxes"][i].tensor.cpu().numpy()[0])],
                          "conf": float(predictions["instances"]._fields["scores"].cpu().numpy()[i])})

    labels_file = os.path.join(directory, '{}.label.tsv'.format(prefix))
    with open(labels_file, 'a') as f:
        text = '{}\t{}\n'.format(idx, json.dumps(labels_info))
        len_label = len_label + len(text)
        f.write(text)

    return len_feature, len_label


# args = get_parser().parse_args()
def make_subset(image_list, caption_file, size, id_dictionary):
    with open(caption_file, 'r') as f:
        captions = json.load(f)
    # swapped_id_dict = dict([(value, key) for key, value in id_dictionary.items()])
    print(type(id_dictionary))
    print(len(id_dictionary))

    swapped_id_dict = {v: k for k, v in id_dictionary.items()}
    print(type(swapped_id_dict))
    print(len(swapped_id_dict))

    n_samples = int(size*len(image_list))
    print('Sampling {} images from total of {}'.format(n_samples, len(image_list)))
    sampled_image_list = random.sample(image_list, n_samples)

    result_captions = []
    for image in sampled_image_list:
        name = image.split('/')[-1]
        img_id = swapped_id_dict[name]
        for caption in captions:
            if caption['image_id'] == img_id:
                result_captions.append(caption)

    print('{} images sampled with {} captions from total of {} captions'.format(len(sampled_image_list), len(result_captions), len(captions)))

    return sampled_image_list, result_captions


def build_feature_dataset(args):
    mp.set_start_method("spawn", force=True)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # prefix = ''+args.input_path.split('/')[-1]

    if args.working_dir is None:
        print('output-features path was not found')
        # TODO: exception?

    if not os.path.isfile(args.coco_classnames):
        print('coco-classnames path was not found')
        # TODO: exception?

    if not os.path.isdir(args.input_path):
        print('input-path path was not found')
        # TODO: exception?

    # n_of_images = (len(glob.glob1(args.input_path, "*.jpg")))
    # if n_of_images > 0:
    #     print('size of image batch: {}'.format(n_of_images))
    # else:
    #     print('no .jpg images in input path found')

    # Create directory in args.working_dir named by size of subset and its number
    src_dir_list = [f.path for f in os.scandir(args.input_path) if f.is_dir()]
    if len(src_dir_list) == 0:
        print('No files found in input directory. train, val or test directory expected.')

    if not os.path.isdir(args.working_dir):
        os.mkdir(args.working_dir)

    if args.data_subset > 0:
        working_dir = os.path.join(args.working_dir, args.input_path.split('/')[-1])
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        subset_dir = os.path.join(working_dir, 'sub{:2.0f}_01'.format(args.data_subset*100))
        if not os.path.isdir(subset_dir):
            os.mkdir(subset_dir)
        else:
            while os.path.isdir(subset_dir):
                subset_dir = subset_dir[:-1] + str(int(subset_dir[-1])+1)
            os.mkdir(subset_dir)
        working_dir = subset_dir
    else:
        working_dir = os.path.join(args.working_dir, args.input_path.split('/')[-1])
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

    if os.path.isfile(args.id_dictionary_file):
        id_dictionary_file = args.id_dictionary_file
    else:
        id_dictionary_file = os.path.join(args.input_path, args.id_dictionary_file)
        assert os.path.isfile(id_dictionary_file)
    with open(id_dictionary_file, 'r') as f:
        id_dictionary = json.load(f)

    # Build
    for directory in src_dir_list:
        prefix = directory.split('/')[-1]
        prepare_files(working_dir, prefix)
        if 'train' in prefix:
            assert os.path.isfile(os.path.join(args.input_path, '{}_caption.json'.format(prefix)))
            caption_file = os.path.join(args.input_path, '{}_caption.json'.format(prefix))
            image_list = [os.path.join(directory, f) for f in os.listdir(directory)]
            image_list, result_captions = make_subset(image_list, caption_file, args.data_subset, id_dictionary)
            with open(os.path.join(working_dir, '{}_caption.json'.format(prefix)), 'w') as f:
                json.dump(result_captions, f)
        else:
            image_list = [os.path.join(directory, f) for f in os.listdir(directory)]

        if 'val' in prefix or 'test' in prefix:
            if not os.path.isfile(os.path.join(args.input_path, '{}_caption.json'.format(prefix))):
                print('{} was not found!'.format(os.path.join(args.input_path, '{}_caption.json'.format(prefix))))
            else:
                shutil.copyfile(os.path.join(args.input_path, '{}_caption.json'.format(prefix)), os.path.join(working_dir, '{}_caption.json'.format(prefix)))

            if not os.path.isfile(os.path.join(args.input_path, '{}_caption_coco_format.json'.format(prefix))):
                print('{} was not found!'.format(os.path.join(args.input_path, '{}_caption_coco_format.json'.format(prefix))))
            else:
                shutil.copyfile(os.path.join(args.input_path, '{}_caption_coco_format.json'.format(prefix)), os.path.join(working_dir, '{}_caption_coco_format.json'.format(prefix)))

        len_feature = 0
        len_label = 0

        id_dictionary_file = os.path.join(working_dir, '{}.id_dictionary.txt'.format(prefix))

        with open(id_dictionary_file, 'w') as f_id:
            pass
        for idx, path in enumerate(image_list):
            with open(id_dictionary_file, 'a') as f_id:
                f_id.write('{}\t{}\n'.format(str(idx), path.split('/')[-1]))
            print('Processing image: {}'.format(path))
            img = read_image(path, format="BGR")
            predictions, visualized_output = demo.run_on_image(img)
            len_feature, len_label = save_features(working_dir, idx, predictions, len_feature, len_label, coco_classnames=args.coco_classnames, prefix=prefix)



if __name__ == "__main__":
    pass
