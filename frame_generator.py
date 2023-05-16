import pathlib

import tensorflow as tf
import random
import cv2
import numpy as np


class FrameGenerator:

    def __init__(self, path: pathlib, n_frames: int, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def format_frame(self, frame, output_size):
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame

    def frames_from_video_file(self, video_path, n_frames=20, output_size=(224, 224), frame_step=15):
        result = []
        src = cv2.VideoCapture(str(video_path))
        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.randint(0, max_start)

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = src.read()
        result.append(self.format_frame(frame, output_size))

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
                if ret:
                    frame = self.format_frame(frame, output_size)
                    result.append(frame)
                else:
                    result.append(np.zeros_like(result[0]))
        src.release()
        result = np.array(result)[..., [2, 1, 0]]
        return result

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob("*/*.avi"))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = self.frames_from_video_file(video_path=path, n_frames=self.n_frames)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, label


# if __name__ == "__main__":
#
#     path = pathlib.Path("data/UCF101_subset/val/")
#     obj = FrameGenerator(path=path, n_frames=10, training=False)
#     # result = obj.frames_from_video_file(video_path=path)
#     frames, label = next(obj())
#     pdb.set_trace()
#     print("Hi")
