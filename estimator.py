""" Estimator model function """
import tensorflow as tf

from model_library import Model


def model_fn(features, labels, mode, params):
    """ Estimator Model Function """

    #####################
    # Input handling
    #####################

    # Extract images and labels tensors
    inputs = features['images']
    labels = {k: features[k] for k in params['label']}

    # Build a lookup table to map output indices to class names
    index_to_class_name_mappings = dict()
    for label in params['label']:
        n_classes = len(params['class_mapping_clean'][label].keys())
        ordered_classes = [params['class_mapping_clean'][label][i]
                           for i in range(0, n_classes)]
        lookup = tf.contrib.lookup.index_to_string_table_from_tensor(
                    tf.constant(ordered_classes),
                    name='%s/class_name_lookup' % label)
        index_to_class_name_mappings[label] = lookup

    # Generate a summary node for the images
    tf.summary.image('images', inputs, max_outputs=6)

    #####################
    # Architecture
    #####################

    # Create model architecture and get outputs
    mod = Model().create(params['model'], inputs,
                         mode == tf.estimator.ModeKeys.TRAIN, params)
    outputs = mod.get_output()

    # create logits for each head
    with tf.name_scope("logits"):
        model_logits = dict()
        for label, n_classes in zip(params['label'], params['n_classes']):
            head_logits = tf.layers.dense(inputs=outputs, units=n_classes,
                                          name=label)
            head_logits = tf.identity(head_logits, label)
            head_logits = tf.cast(head_logits, tf.float32, name=label)
            model_logits[label] = head_logits

    #####################
    # Predictions
    #####################

    # Create Prediction Outputs
    with tf.name_scope("predictions"):
        predictions = dict()
        for label, logits in model_logits.items():
            k_for_topk = tf.reduce_min([tf.shape(logits)[1],
                                        tf.constant(5)])
            table = index_to_class_name_mappings[label]

            # Logits
            top_k_values, top_k_indices = tf.nn.top_k(logits, k=k_for_topk)
            top_1_values, top_1_indices = tf.nn.top_k(logits, k=1)

            # Probs
            probs = tf.nn.softmax(logits, name='softmax_tensor')
            top_1_probs, _ = tf.nn.top_k(probs, k=1)
            top_k_probs, _ = tf.nn.top_k(probs, k=k_for_topk)

            preds = {
              '%s/top_1_class' % label: top_1_indices,
              '%s/top_1_class_props' % label: top_1_probs,
              '%s/top_1_class_name' % label: table.lookup(
                tf.cast(top_1_indices, tf.int64)),
              '%s/top_k_class' % label: top_k_indices,
              '%s/top_k_class_names' % label: table.lookup(
                tf.cast(top_k_indices, tf.int64)),
              '%s/top_k_class_probs' % label: top_k_probs,
              '%s/probabilities' % label: probs
              }
            predictions = {**predictions, **preds}

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
                })

    #####################
    # Loss and Optimizer
    #####################

    # Calculate loss
    with tf.name_scope("cross_entropy"):
        for label in params['label']:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    logits=model_logits[label], labels=labels[label],
                    reduction=tf.losses.Reduction.MEAN)
            cross_entropy = tf.identity(cross_entropy, name=label)
            tf.losses.add_loss(cross_entropy)

    # Sum all losses in the loss collection (alternatively, use the mean to
    # compensate for the imbalance wrt. the reg. loss)
    loss = tf.losses.get_total_loss(
        add_regularization_losses=False,
        name='loss')

    # Get sum of all regularization losses
    reg_loss = tf.losses.get_regularization_loss(name='regularization_loss')
    tf.summary.scalar('reg_loss', reg_loss)

    total_loss = tf.add_n([loss, reg_loss], 'total_loss')
    tf.summary.scalar('total_loss', total_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # TODO: Dynamic learning rate and momentum optimizer
        # learning_rate = tf.constant(0.01)
        #
        # # Create a tensor named learning_rate for logging purposes
        # tf.identity(learning_rate, name='learning_rate')
        # tf.summary.scalar('learning_rate', learning_rate)

        # optimizer = tf.train.MomentumOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=params['momentum']
        #     )
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.01,
            decay=0.9,
            momentum=params['momentum'],
            epsilon=1.0)

        minimize_op = optimizer.minimize(total_loss, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    ########################
    # Metrics and Summaries
    #######################

    # Calculate Metrics
    def create_summary(metrics, metric, label):
        """ Create Summary Node """
        tf.identity(metrics['%s/%s' % (label, metric)],
                    name='%s/train_%s' % (label, metric))
        tf.summary.scalar('%s/train_%s' % (label, metric),
                          metrics['%s/%s' % (label, metric)][1])

    def create_per_recall_class_summary():
        pass

    with tf.name_scope("metrics"):
        metrics = dict()
        for label, n_classes in zip(params['label'], params['n_classes']):
            y_true = labels[label]
            y_pred = predictions['%s/top_1_class' % label]

            mets = {
                '%s/accuracy' % label: tf.metrics.accuracy(y_true, y_pred),
                '%s/precision' % label: tf.metrics.precision(y_true, y_pred),
                '%s/recall' % label: tf.metrics.recall(y_true, y_pred)
                }
            metrics = {**metrics, **mets}

            # Create Summary Tensors
            create_summary(metrics, 'accuracy', label)
            create_summary(metrics, 'precision', label)
            if n_classes == 2:
                create_summary(metrics, 'recall', label)

    eval_metric_ops = {'metrics/%s' % k: v for k, v in metrics.items()}

    if mode == tf.estimator.ModeKeys.TRAIN:

        return tf.estimator.EstimatorSpec(
           mode=mode,
           predictions=predictions,
           loss=loss,
           train_op=train_op,
           eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.EVAL:
        # Does not work properly

        # display misclassifications / label
        # for label in params['label']:
        #     y_true = labels[label]
        #     y_pred = predictions['%s/top_1_class' % label]
        #     wrong_idx = tf.not_equal(y_true, y_pred)
        #     wrong_preds = tf.boolean_mask(inputs, wrong_idx, axis=0)
        #     # n_to_show = tf.reshape(tf.minimum(tf.shape(wrong_preds)[0], 6), [])
        #
        #     # Generate a summary node for misclassifications
        #     tf.summary.image('misclassifications_%s' % label,
        #                      wrong_preds, max_outputs=6)
        #
        # # Create a SummarySaverHook
        # summary_hook = tf.train.SummarySaverHook(
        #             save_steps=1,
        #             output_dir="./eval",
        #             summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=predictions,
                        loss=loss,
                        train_op=train_op,
                        eval_metric_ops=eval_metric_ops)
