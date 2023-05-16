import pathlib
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import tensorflow as tf
from frame_generator import FrameGenerator


class TrainVideoModel:

    def __init__(self, n_frames, batch_size):
        self.n_frames = n_frames
        self.train_data_path = pathlib.Path("sample_code/data/UCF101_subset/train")
        self.val_data_path = pathlib.Path("sample_code/data/UCF101_subset/val")
        self.batch_size = batch_size
        pass

    def batch_generator(self, path, training=False):

        output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.int16))
        ds = tf.data.Dataset.from_generator(FrameGenerator(path=path, n_frames=self.n_frames, training=training),
                                            output_signature=output_signature)
        # ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(self.batch_size)

        return ds

    def dataset_creator(self):

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = self.batch_generator(self.train_data_path)
        # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        val_ds = self.batch_generator(self.val_data_path)
        # val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds

    def build_classifier(self, resolution, model_id, num_classes):
        """Builds a classifier on top of a backbone model."""
        tf.keras.backend.clear_session()
        backbone = movinet.Movinet(model_id=model_id)
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=num_classes)
        model.build([self.batch_size, self.n_frames, resolution, resolution, 3])

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

        return model


    def train_model(self, num_classes):

        train_ds, val_ds = self.dataset_creator()
        model = self.build_classifier(resolution=224, model_id="a0", num_classes=num_classes)
        model.fit(train_ds, validation_data=val_ds, epochs=2, validation_freq=1, verbose=1)

        return model
