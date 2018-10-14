# This is a helper module that contains conveniences to access the MS COCO
# dataset. You can modify at will.  In fact, you will almost certainly have
# to, or implement otherwise.

import sys

# This is evil, forgive me, but practical under the circumstances.
# It's a hardcoded access to the COCO API.  
COCOAPI_PATH = '/scratch/lt2316-h18-resources/cocoapi/PythonAPI/'
TRAIN_ANNOTATION_FILE = '/scratch/lt2316-h18-resources/coco/annotations/instances_train2017.json'
VALIDATION_ANNOTATION_FILE = '/scratch/lt2316-h18-resources/coco/annotations/instances_val2017.json'
TRAIN_CAPTION_FILE = '/scratch/lt2316-h18-resources/coco/annotations/captions_train2017.json'
VAL_CAPTION_FILE = '/scratch/lt2316-h18-resources/coco/annotations/captions_val2017.json'
TRAIN_IMAGE_DIR = '/scratch/lt2316-h18-resources/coco/train2017/'
VALIDATION_IMAGE_DIR = '/scratch/lt2316-h18-resources/coco/val2017/'
annotation_file = TRAIN_ANNOTATION_FILE
caption_file = TRAIN_CAPTION_FILE
image_directory = TRAIN_IMAGE_DIR

sys.path.append(COCOAPI_PATH)
from pycocotools.coco import COCO

annotation_coco = None
caption_coco = None
category_dictionary = {}

# OK back to normal.
import random
import skimage.io as io
import skimage.transform as tform
import numpy as np


def setmode(mode):
    """
    Set entire module's mode as 'train' or 'test' for the purpose of data extraction.
    """
    global annotation_file
    global caption_file
    global image_directory
    global annotation_coco, caption_coco
    global category_dictionary
    if mode == "train":
        annotation_file = TRAIN_ANNOTATION_FILE
        caption_file = TRAIN_CAPTION_FILE
        image_directory = TRAIN_IMAGE_DIR
    elif mode == "test":
        annotation_file = VALIDATION_ANNOTATION_FILE
        caption_file = VAL_CAPTION_FILE
        image_directory = VALIDATION_IMAGE_DIR
    else:
        raise ValueError

    annotation_coco = COCO(annotation_file)
    caption_coco = COCO(caption_file)

    # To facilitate category lookup.
    cats = annotation_coco.getCatIds()
    category_dictionary = {x: (annotation_coco.loadCats(ids=[x])[0]['name']) for x in cats}


def query(queries, exclusive=True):
    """
    Collects mutually-exclusive lists of COCO ids by queries, so returns
    a parallel list of lists.
    (Setting 'exclusive' to False makes the lists non-exclusive.)
    e.g., exclusive_query([['toilet', 'boat'], ['umbrella', 'bench']])
    to find two mutually exclusive lists of images, one with toilets and
    boats, and the other with umbrellas and benches in the same image.
    """
    if not annotation_coco:
        raise ValueError
    image_sets = [set(annotation_coco.getImgIds(catIds=annotation_coco.getCatIds(catNms=x))) for x in queries]
    if len(queries) > 1:
        if exclusive:
            common = set.intersection(*image_sets)
            return [[x for x in y if x not in common] for y in image_sets]
        else:
            return [list(y) for y in image_sets]
    else:
        return [list(image_sets[0])]


def get_captions_for_ids(id_list):
    annotation_ids = caption_coco.getAnnIds(imgIds=id_list)
    annotations = caption_coco.loadAnns(annotation_ids)
    return [ann['caption'] for ann in annotations]


def get_cats_for_img(image_id):
    """
    Takes an image id and gets a category list for it.
    """
    if not annotation_coco:
        raise ValueError

    image_nn_ids = annotation_coco.getAnnIds(imgIds=image_id)
    image_nns = annotation_coco.loadAnns(image_nn_ids)
    return list(set([category_dictionary[x['category_id']] for x in image_nns]))


def iter_captions(id_lists, categories, batch=1):
    """
    Obtains the corresponding captions from multiple COCO id lists.
    Randomizes the order.
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    """
    if not caption_coco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(id_lists, categories):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        random_list = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in random_list:
            annotation_ids = caption_coco.getAnnIds(imgIds=[p[0]])
            annotations = caption_coco.loadAnns(annotation_ids)
            for ann in annotations:
                captions.append(ann['caption'])
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []


def iter_captions_cats(id_lists, cats, batch=1):
    """
    Obtains the corresponding captions from multiple COCO id lists alongside all associated image captions per image.
    Randomizes the order.
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    """
    if not caption_coco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(id_lists, cats):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        random_list = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in random_list:
            annotation_ids = caption_coco.getAnnIds(imgIds=[p[0]])
            annotations = caption_coco.loadAnns(annotation_ids)
            for ann in annotations:
                image_id = ann['image_id']
                cats = get_cats_for_img(image_id)
                captions.append((ann['caption'], cats))
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []


def iter_images(id_lists, cats, size=(200, 200), batch=1):
    """
    Obtains the corresponding image data as numpy array from multiple COCO id lists.
    Returns an infinite iterator (do not convert to list!) that returns tuples (imagess, categories)
    as parallel lists at size of batch.
    By default, randomizes the order and resizes the image.
    """
    if not annotation_coco:
        raise ValueError
    if batch < 1:
        raise ValueError
    if not size:
        raise ValueError  # size is mandatory

    full = []
    for z in zip(id_lists, cats):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        random_list = random.sample(full, k=len(full))

        images = []
        labels = []
        for r in random_list:
            image_file = annotation_coco.loadImgs([r[0]])[0]['file_name']
            image = io.imread(image_directory + image_file)
            image_scaled = tform.resize(image, size)
            # Colour images only.
            if image_scaled.shape == (size[0], size[1], 3):
                images.append(image_scaled)
                labels.append(r[1])
                if len(images) % batch == 0:
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []
