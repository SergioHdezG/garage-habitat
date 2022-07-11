#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
import cv2

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config("configs/tasks/pointnav.yaml")
    ) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
            count_steps += 1
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
            cv2.waitKey(0)
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
