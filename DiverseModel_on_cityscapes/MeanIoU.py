import tensorflow as tf


class MIoU(tf.keras.metrics.MeanIoU):
    def __init__(
            self,
            num_classes: int,
            name=None,
            dtype=None,
    ):
        super(MIoU, self).__init__(
            num_classes=num_classes,
            name='MIoU',
            dtype=dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)
