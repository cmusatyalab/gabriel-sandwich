# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#           Roger Iyengar <iyengar@cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import os
import math
import instruction_pb2
from gabriel_protocol import gabriel_pb2
from collections import namedtuple

ENGINE_NAME = "instruction"

# The following code outputs [7, 4, 8, 1, 3, 2, 5, 6, 0]:
# # build a mapping between faster-rcnn recognized object order to a
# # standard order
# LABELS = ["bread", "ham", "cucumber", "lettuce", "cheese", "half", "hamwrong",
#           "tomato", "full"]
# self._object_mapping = [-1] * len(LABELS)
# with open(os.path.join('model', 'labels.txt')) as f:
#     lines = f.readlines()
#     for idx, line in enumerate(lines):
#         line = line.strip()
#         self._object_mapping[idx] = LABELS.index(line)
# print(self._object_mapping)

# bread is label 0. 0 is in position 8 in the list. Therefore, BREAD = 8 + 1
# ham is label 1. 1 is in position 3 in the list. Therefor, HAM = 3 + 1
TOMATO = 1
CHEESE = 2
FULL = 3
HAM = 4
LETTUCE = 5
CUCUMBER = 6
HALF = 7
HAMWRONG = 8
BREAD = 9

DONE = 10  # This is not a class the DNN will output

Hologram = namedtuple('Hologram', ['dist', 'x', 'y', 'label_index'])

HAM_HOLO = Hologram(dist=6500, x=0.5, y=0.36)
LETTUCE_HOLO = Hologram(dist=6800, x=0.5, y=0.32)
BREAD_HOLO = Hologram(dist=7100, x=0.5, y=0.3)
TOMATO_HOLO = Hologram(dist=7500, x=0.5, y=0.26)
BREAD_TOP_HOLO = Hologram(dist=7800, x=0.5, y=0.22)

INSTRUCTIONS = {
    BREAD: 'Now put a piece of bread on the table.',
    HAM: 'Now put a piece of ham on the bread.',
    LETTUCE: 'Now put a piece of lettuce on the ham.',
    HALF: 'Now put a piece of bread on the lettuce.',
    CUCUMBER: 'This sandwich doesn\'t contain any cucumber. Replace the '
              'cucumber with lettuce.',
    HAMWRONG: 'That\'s too much meat. Replace the ham with tomatoes.'
    TOMATO: 'You are half done. Now put a piece of tomato on the bread.',
    FULL: 'Now put the bread on top and you will be done.',
    DONE: 'Congratulations! You have made a sandwich!',
}

IMAGE_FILENAMES = {
    BREAD: 'bread.jpeg',
    HAM: 'ham.jpeg',
    LETTUCE: 'lettuce.jpeg',
    HALF: 'half.jpeg',
    CUCUMBER: 'lettuce.jpeg'
    HAMWRONG: 'tomato.jpeg',
    TOMATO: 'tomato.jpeg',
    FULL: 'full.jpeg',
    DONE: 'full.jpeg',
}

IMAGES = {
    class_idx: open(os.path.join('images_feedback', filename), 'rb').read()
    for class_idx, filename in IMAGE_FILENAMES.items()
}


def _result_without_update(engine_fields):
    result_wrapper = gabriel_pb2.ResultWrapper()
    result_wrapper.engine_fields.Pack(engine_fields)
    return result_wrapper


def _result_with_update(engine_fields, class_idx):
    engine_fields.update_count += 1

    result_wrapper = _result_without_update(engine_fields)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.IMAGE
    result.engine_name = ENGINE_NAME
    result.payload = IMAGES[class_idx]
    result_wrapper.results.append(result)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.engine_name = ENGINE_NAME
    result.payload = INSTRUCTIONS[class_idx].encode(encoding="utf-8")
    result_wrapper.results.append(result)

    return result_wrapper


def _start_result(engine_fields):
    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.NOTHING
    return _result_with_update(BREAD)


def _nothing_result(det_for_class, engine_fields):
    if BREAD not in det_for_class:
        return _result_without_update(engine_fields)

    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.BREAD
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater._update_holo_location(det_for_class[BREAD], HAM_HOLO)

    return _result_with_update(engine_fields, HAM)


def _bread_result(det_for_class, engine_fields):
    if HAM not in det_for_class:
        if BREAD in det_for_class:

            # We have to increase this so the client will process the hologram
            # update
            engine_fields.update_count += 1

            hologram_updater = _HologramUpdater(engine_fields)
            hologram_updater._update_holo_location(
                det_for_class[BREAD], HAM_HOLO)
        return _result_without_update(engine_fields)

    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.HAM
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater._update_holo_location(det_for_class[HAM], LETTUCE_HOLO)

    return _result_with_update(engine_fields, LETTUCE)


def _lettuce_helper(det_for_class, engine_fields):
    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.LETTUCE
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater._update_holo_location(det_for_class[LETTUCE], BREAD_HOLO)
    return _result_with_update(engine_fields, HALF)


def _ham_result(det_for_class, engine_fields):
    if LETTUCE in det_for_class:
        return _lettuce_helper(det_for_class, engine_fields)
    elif CUCUMBER in det_for_class:
        engine_fields.sandwich.state = instruction_pb2.Sandwich.State.CUCUMBER
        return _result_with_update(engine_fields, CUCUMBER)
    elif (HAM not in det_for_class) and (BREAD in det_for_class):
        return _nothing_result(det_for_class, engine_fields)

    if HAM in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater._update_holo_location(det_for_class[HAM], LETTUCE_HOLO)
    return _result_without_update(engine_fields)


def _half_helper(det_for_class, engine_fields):
    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.HALF
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater._update_holo_location(det_for_class[HALF], TOMATO_HOLO)
    return _result_with_update(engine_fields, TOMATO)


def _lettuce_result(det_for_class, engine_fields):
    if HALF in det_for_class:
        return _half_helper(det_for_class, engine_fields)
    elif (HAM in det_for_class) and (LETTUCE not in det_for_class):
        # Put us in the ham state
        return _bread_result(det_for_class, engine_fields)

    if LETTUCE in det_for_class:
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater._update_holo_location(
            det_for_class[LETTUCE], BREAD_HOLO)
    return _result_without_update(engine_fields)


def _cucumber_result(det_for_class, engine_fields):
    if LETTUCE in det_for_class:
        return _lettuce_helper(det_for_class, engine_fields)
    elif (HAM in det_for_class) and (LETTUCE not in det_for_class):
        # Put us in the ham state
        return _bread_result(det_for_class, engine_fields)

    return _result_without_update(engine_fields)


def _tomato_helper(det_for_class, engine_fields):
    engine_fields.sandwich.state = instruction_pb2.Sandwich.State.TOMATO
    hologram_updater = _HologramUpdater(engine_fields)
    hologram_updater._update_holo_location(
        det_for_class[TOMATO], BREAD_TOP_HOLO)
    return _result_with_update(engine_fields, FULL)


def _half_result(det_for_class, engine_fields):
    if TOMATO in det_for_class:
        return _tomato_helper(det_for_class, engine_fields)
    elif HAMWRONG in det_for_class:
        engine_fields.sandwich.state = instruction_pb2.Sandwich.State.HAM_WRONG
        return _result_with_update(engine_fields, HAMWRONG)
    elif (LETTUCE in det_for_class) and (HALF not in det_for_class):
        # Update state to lettuce.
        return _lettuce_helper(det_for_class, engine_fields)

    if HALF in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater._update_holo_location(det_for_class[HALF], TOMATO_HOLO)
    return _result_without_update(engine_fields)


def _tomato_result(det_for_class, engine_fields):
    if FULL in det_for_class:
        engine_fields.sandwich.state = instruction_pb2.Sandwich.State.FULL
        return _result_with_update(engine_fields, DONE)
    elif (HALF in det_for_class) and (TOMATO not in det_for_class):
        # Update state to Half
        return _half_helper(det_for_class, engine_fields)

    if TOMATO in det_for_class:
        engine_fields.update_count += 1
        hologram_updater = _HologramUpdater(engine_fields)
        hologram_updater._update_holo_location(
            det_for_class[TOMATO], BREAD_TOP_HOLO)
    return _result_without_update(engine_fields)


def _ham_wrong_result(det_for_class, engine_fields):
    if TOMATO in det_for_class:
        return _tomato_helper(det_for_class, engine_fields)
    elif HALF in det_for_class:
        return _half_helper(det_for_class, engine_fields)

    return _result_without_update(engine_fields)


class _HologramUpdater:
    def __init__(self, engine_fields):
        self._engine_fields = engine_fields

    def update_holo_location(self, det, holo):
        x1, y1, x2, y2 = det[:4]
        x = x1 * (1 - holo.x) + x2 * holo.x
        y = y1 * (1 - holo.y) + y2 * holo.y
        area = (y2 - y1) * (x2 - x1)

        depth = math.sqrt(holo.dist / float(area))
        self._engine_fields.sandwich.holo_x = x
        self._engine_fields.sandwich.holo_y = y
        self._engine_fields.sandwich.holo_depth = depth


def get_instruction(engine_fields, det_for_class):
    state = engine_fields.sandwich.state

    if state == instruction_pb2.Sandwich.State.START:
        return _start_result(engine_fields)

    if len(det_for_class) < 1:
        return _result_without_update(engine_fields)

    if state == instruction_pb2.Sandwich.State.NOTHING:
        return _nothing_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.BREAD:
        return _bread_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.HAM:
        return _ham_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.LETTUCE:
        return _lettuce_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.CUCUMBER:
        return _cucumber_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.HALF:
        return _half_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.TOMATO:
        return _tomato_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.HAM_WRONG:
        return _ham_wrong_result(det_for_class, engine_fields)
    elif state == instruction_pb2.Sandwich.State.FULL:
        return _result_without_update(engine_fields)

    raise Exception("Invalid state")
