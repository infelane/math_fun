import tensorflow as tf

_EPSILON = 10e-8


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


# TODO with weight!
def weighted_categorical_crossentropy(output, target, weight, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        target: A tensor of the same shape as `output`.
        weight: A list with the weights to scale the cost with
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        weight = _to_tensor(weight, output.dtype.base_dtype)
        # TODO check if multiplied to right values! otherwise transform weight to shape (1, ..., 1, n_out)
        # TODO although it seems to work
        return - tf.reduce_sum(weight * target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        # TODO
        raise NotImplementedError
        # return tf.nn.softmax_cross_entropy_with_logits(labels=target,
        #                                                logits=output)
