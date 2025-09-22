# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""


# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": (250, 200, 100), "isthing": 1, "id": 1, "name": "vehicle"},
    {"color": (220, 20, 60), "isthing": 1, "id": 2, "name": "pedestrian"},

]

# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

# All Cityscapes categories, together with their nice-looking visualization colors
# It's from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  # noqa
CITYSCAPES_CATEGORIES = [
    {"color": (0, 102, 0), "isthing": 0, "id": 0, "trainId": 0, "name": "runway"},
    {"color": (128, 64, 128), "isthing": 0, "id": 1, "trainId": 1, "name": "road"},
    {"color": (0, 100, 150), "isthing": 0, "id": 2, "trainId": 2, "name": "mat"},
    {"color": (255, 255, 0), "isthing": 0, "id": 3, "trainId": 3, "name": "braille-block"},
    {"color": (64, 255, 0), "isthing": 0, "id": 4, "trainId": 4, "name": "cross-walk"},
    {"color": (204, 51, 204), "isthing": 0, "id": 5, "trainId": 5, "name": "bicycle-road"},
    {"color": (153, 102, 0), "isthing": 0, "id": 6, "trainId": 6, "name": "speed-bump"},
    {"color": (152, 251, 152), "isthing": 0, "id": 7, "trainId": 7, "name": "terrain"},
    {"color": (107, 142, 35), "isthing": 0, "id": 8, "trainId": 8, "name": "vegetation"},
    {"color": (70, 130, 180), "isthing": 0, "id": 9, "trainId": 9, "name": "sky"},
    {"color": (250, 0, 150), "isthing": 0, "id": 10, "trainId": 10, "name": "pole"},
    {"color": (116, 27, 71), "isthing": 0, "id": 11, "trainId": 11, "name": "pole-group"},
    {"color": (220, 20, 60), "isthing": 1, "id": 12, "trainId": 12, "name": "pedestrian"},
    {"color": (64, 0, 128), "isthing": 0, "id": 13, "trainId": 13, "name": "traffic-sign"},
    {"color": (250, 170, 30), "isthing": 0, "id": 14, "trainId": 14, "name": "traffic-light-front"},
    {"color": (153, 0, 255), "isthing": 0, "id": 15, "trainId": 15, "name": "traffic-light-back"},
    {"color": (190, 153, 153), "isthing": 0, "id": 16, "trainId": 16, "name": "fence"},
    {"color": (250, 200, 100), "isthing": 1, "id": 17, "trainId": 17, "name": "vehicle"},
    {"color": (150, 100, 255), "isthing": 0, "id": 18, "trainId": 18, "name": "dynamic"},
    {"color": (0, 128, 128), "isthing": 0, "id": 19, "trainId": 19, "name": "static"},
    {"color": (255, 0, 0), "isthing": 0, "id": 20, "trainId": 255, "name": "forbidden-area"},
    {"color": (63, 63, 63), "isthing": 0, "id": 21, "trainId": 255, "name": "void"},
]

# fmt: off
ADE20K_SEM_SEG_CATEGORIES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road, route", "bed", "window ", "grass", "cabinet", "sidewalk, pavement", "person", "earth, ground", "door", "table", "mountain, mount", "plant", "curtain", "chair", "car", "water", "painting, picture", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "tub", "rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove", "palm, palm tree", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus", "towel", "light", "truck", "tower", "chandelier", "awning, sunshade, sunblind", "street lamp", "booth", "tv", "plane", "dirt track", "clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "pool", "stool", "barrel, cask", "basket, handbasket", "falls", "tent", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light", "tray", "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass, drinking glass", "clock", "flag", # noqa
]
# After processed by `prepare_ade20k_sem_seg.py`, id 255 means ignore
# fmt: on


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 2, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 0, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_panoptic_standard":
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in COCO_CATEGORIES]
        thing_colors = [k["color"] for k in COCO_CATEGORIES]
        stuff_classes = [k["name"] for k in COCO_CATEGORIES]
        stuff_colors = [k["color"] for k in COCO_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(COCO_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "vehicle", "pedestrian", # 2
        ]
        CITYSCAPES_STUFF_CLASSES = [
            'runway', 'road', 'mat', 'braille-block', 'cross-walk', 'bicycle-road', # 6
            'speed-bump', 'dynamic', 'static', 'void', 'terrain', # 5
            'vegetation', 'sky', 'pole', 'pole-group', 'traffic-sign', 'traffic-light-front', # 6
            'traffic-light-back', 'fence', # 2
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
