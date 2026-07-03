##########################################################################
##                                                                      ##
##  Copyright (c) 2019 Philipp Lösel. All rights reserved.              ##
##                                                                      ##
##  This file is part of the open source project biomedisa.             ##
##                                                                      ##
##  Licensed under the European Union Public Licence (EUPL)             ##
##  v1.2, or - as soon as they will be approved by the                  ##
##  European Commission - subsequent versions of the EUPL;              ##
##                                                                      ##
##  You may redistribute it and/or modify it under the terms            ##
##  of the EUPL v1.2. You may not use this work except in               ##
##  compliance with this Licence.                                       ##
##                                                                      ##
##  You can obtain a copy of the Licence at:                            ##
##                                                                      ##
##  https://joinup.ec.europa.eu/page/eupl-text-11-12                    ##
##                                                                      ##
##  Unless required by applicable law or agreed to in                   ##
##  writing, software distributed under the Licence is                  ##
##  distributed on an "AS IS" basis, WITHOUT WARRANTIES                 ##
##  OR CONDITIONS OF ANY KIND, either express or implied.               ##
##                                                                      ##
##  See the Licence for the specific language governing                 ##
##  permissions and limitations under the Licence.                      ##
##                                                                      ##
##########################################################################

import keras
from keras import ops
import tensorflow as tf

def model_fit(model, strategy, training_generator, callbacks, epochs, initial_epoch, nb_labels, channels, pseudo_labels=False):

    def gen():
        for i in range(len(training_generator)):
            yield training_generator[i]

    @tf.function
    def distributed_step(step_fn, batch):
        return strategy.run(
            step_fn,
            args=(batch,)
        )

    for cb in callbacks:
        cb.set_model(model)
        cb.set_params({
            "epochs": epochs,
            "steps": len(training_generator),
            "verbose": 1,
        })

    for cb in callbacks:
        cb.on_train_begin()

    if pseudo_labels:
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "x_l": tf.TensorSpec(shape=(None, *(64, 64, 64, channels)), dtype=tf.float32),
                "y_l": tf.TensorSpec(shape=(None, *(64, 64, 64, nb_labels)), dtype=tf.float32),
                "x_u": tf.TensorSpec(shape=(None, *(64, 64, 64, channels)), dtype=tf.float32),
            }
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "x_l": tf.TensorSpec(shape=(None, *(64, 64, 64, channels)), dtype=tf.float32),
                "y_l": tf.TensorSpec(shape=(None, *(64, 64, 64, nb_labels)), dtype=tf.float32),
                "x_u": tf.TensorSpec(shape=(None, *(9, 64, 64, 64, channels)), dtype=tf.float32),
            }
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    for epoch in range(initial_epoch, epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        model.current_epoch.assign(epoch + 1)

        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        # choose step function
        if epoch+1 <= 12:
            step_fn = model.train_step_sup
        elif pseudo_labels:
            step_fn = model.train_step_pseudo
        else:
            step_fn = model.train_step_full

        # ---- metric accumulators (like fit()) ----
        loss_sum = 0.0
        sup_sum = 0.0
        unsup_sum = 0.0
        n_batches = 0

        from tqdm import tqdm
        pbar = tqdm(range(len(training_generator)))
        it = iter(dist_dataset)
        #with strategy.scope(): #TODO: probably not required
        for step in pbar:

            for cb in callbacks:
                cb.on_train_batch_begin(step)

            batch = next(it)
            logs = distributed_step(step_fn, batch)

            #batch = training_generator[step]
            #logs = step_fn(batch)

            logs = {
                k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                for k, v in logs.items()
            }
            logs = {k: float(v.numpy()) for k, v in logs.items()}
            pbar.set_postfix(loss=f"{logs['loss']:.4f}")

            for cb in callbacks:
                cb.on_train_batch_end(step, logs)

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)

        if hasattr(training_generator, "on_epoch_end"):
            training_generator.on_epoch_end()

    for cb in callbacks:
        cb.on_train_end()

def consistency_loss(p1, p2):
    return ops.mean((p1 - p2) ** 2)

def kl_loss(p1, p2):
    eps = 1e-6
    conf = tf.reduce_max(p1, axis=-1, keepdims=True)
    # soft weighting instead of hard mask
    weight = tf.clip_by_value((conf - 0.9) / 0.1, 0.0, 1.0)
    kl = p1 * tf.math.log((p1 + eps) / (p2 + eps))
    #return ops.sum(weight * kl) / (ops.sum(weight) + eps)

    # sum over spatial + channel dims → keep batch
    weighted_kl = weight * kl
    numerator = tf.reduce_sum(weighted_kl, axis=[1,2,3,4])
    denominator = tf.reduce_sum(weight, axis=[1,2,3,4]) + eps

    return numerator / denominator   # shape: (B,)

def extract_overlap(p1, p2, shift):
    """
    shift: (dx, dy, dz) where each is -1, 0, or 1
           indicating relative position of p2 w.r.t p1
    """
    slices_p1 = []
    slices_p2 = []

    for d in shift:
        if d == 1:  # p2 is in + direction
            slices_p1.append(slice(32, 64))
            slices_p2.append(slice(0, 32))
        elif d == -1:  # p2 is in - direction
            slices_p1.append(slice(0, 32))
            slices_p2.append(slice(32, 64))
        else:  # same position (should not really happen for neighbors)
            slices_p1.append(slice(0, 64))
            slices_p2.append(slice(0, 64))

    o1 = p1[:, slices_p1[0], slices_p1[1], slices_p1[2], :]
    o2 = p2[:, slices_p2[0], slices_p2[1], slices_p2[2], :]

    return o1, o2

class SemiSupervisedModel(keras.Model):
    def __init__(self, student, teacher, lambda_consistency=0.01, batch_size=24,  **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.lambda_consistency = lambda_consistency
        self.batch_size = batch_size
        self.ce = keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.val_dice = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.current_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, x, training=False):
        if isinstance(x, dict):
            x = x["x_l"]
        return self.student(x, training=training)

    def ramp_weight(self, epoch, ramp_epochs=20):
        '''epoch = ops.cast(epoch, "float32")
        ramp_epochs = ops.cast(ramp_epochs, "float32")
        alpha = (epoch-12) / (ramp_epochs + 1e-6)
        alpha = ops.clip(alpha, 0.0, 1.0)'''
        alpha = (epoch-12) / (ramp_epochs)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        return self.lambda_consistency * alpha

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        loss = ops.mean(self.ce(y, y_pred))
        return {"val_loss": loss}

    @tf.function
    def train_step_sup(self, data):
        x_l = data["x_l"]
        y_l = data["y_l"]

        with tf.GradientTape() as tape:
            y_pred = self.student(x_l, training=True)
            per_voxel_loss = self.ce(y_l, y_pred)
            per_example_loss = tf.reduce_mean(
                per_voxel_loss,
                axis=[1,2,3]  # reduce spatial dims only
            )
            loss = tf.nn.compute_average_loss(
                per_example_loss,
                global_batch_size=self.batch_size
            )

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        return {
            "loss": loss,
            "supervised": loss,
            "unsupervised": tf.constant(0.0, tf.float32),
        }

    @tf.function
    def train_step_full(self, data):
        x_l = data["x_l"]
        y_l = data["y_l"]
        x_u = data["x_u"]

        with tf.GradientTape() as tape:

            # supervised
            y_pred_l = self.student(x_l, training=True)
            per_voxel_loss = self.ce(y_l, y_pred_l)
            per_example_loss = tf.reduce_mean(
                per_voxel_loss,
                axis=[1,2,3]  # reduce spatial dims only
            )
            loss_sup = tf.nn.compute_average_loss(
                per_example_loss,
                global_batch_size=self.batch_size
            )

            # --- UNSUPERVISED (heavy) ---
            B = tf.shape(x_u)[0]
            num_patches = 9

            x_u_flat = tf.reshape(x_u, (B * num_patches, *x_u.shape[2:]))
            y_pred_u = self.student(x_u_flat, training=True)
            y_pred_u = tf.reshape(y_pred_u, (B, num_patches, *y_pred_u.shape[1:]))

            cons = tf.zeros((B,), tf.float32)
            count = 0

            p1 = y_pred_u[:, 0]

            for i, shift in enumerate([
                (-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1),
                (-1,1,1),(1,-1,1),(1,1,-1),(1,1,1)
            ]):
                p2 = y_pred_u[:, i+1]
                p2 = tf.stop_gradient(p2)

                o1, o2 = extract_overlap(p1, p2, shift)
                kl = kl_loss(o2, o1)              # expect (B, ...)

                cons += kl
                count += 1

            per_example_unsup = cons / (count + 1e-6)

            loss_unsup = tf.nn.compute_average_loss(
                per_example_unsup,
                global_batch_size=self.batch_size #TODO: batch_size*4?
            )

            '''cons = tf.constant(0.0, tf.float32)
            count = 0

            p1 = y_pred_u[:, 0]
            for i, shift in enumerate([
                (-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1),
                (-1,1,1),(1,-1,1),(1,1,-1),(1,1,1)
            ]):
                p2 = y_pred_u[:, i+1]
                p2 = tf.stop_gradient(p2)
                o1, o2 = extract_overlap(p1, p2, shift)
                cons += kl_loss(o2, o1)
                count += 1

            loss_unsup = cons / (count + 1e-6)'''

            loss = loss_sup + self.ramp_weight(self.current_epoch) * loss_unsup

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        return {
            "loss": loss,
            "supervised": loss_sup,
            "unsupervised": loss_unsup,
        }

    '''def train_step(self, data):
        # unpack
        x_l = data["x_l"]
        y_l = data["y_l"]
        x_u = data["x_u"]

        with tf.GradientTape() as tape:

            # --- supervised branch ---
            y_pred_l = self.student(x_l, training=True)
            loss_sup = ops.mean(self.ce(y_l, y_pred_l))

            # --- unsupervised branch ---
            # x_u shape: (B, 9, 64, 64, 64, C)
            B = ops.shape(x_u)[0]
            num_patches = 9

            # flatten patches
            x_u_flat = ops.reshape(x_u, (B * num_patches, *x_u.shape[2:]))
            y_pred_u = self.student(x_u_flat, training=True)
            y_pred_u = ops.reshape(y_pred_u, (B, num_patches, *y_pred_u.shape[1:]))

            # consistency across patch pairs
            cons = 0.0
            count = 0
            p1 = y_pred_u[:, 0]
            for i, shift in enumerate([(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1),(-1,1,1),(1,-1,1),(1,1,-1),(1,1,1)]):
                p2 = y_pred_u[:, i+1]
                p2 = ops.stop_gradient(p2)
                o1, o2 = extract_overlap(p1, p2, shift)
                cons += kl_loss(o2, o1)
                #cons += consistency_loss(o1, o2)
                count += 1

            # unsupervised loss
            loss_unsup = cons / (count + 1e-6)

            # --- total ---
            #lambda_ssl = self.lambda_consistency * tf.sigmoid((self.val_dice - 0.7) * 20)
            loss = loss_sup + self.ramp_weight(self.current_epoch) * loss_unsup

        # gradients
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        return {
            "loss": loss,
            "supervised": loss_sup,
            "unsupervised": loss_unsup,
        }'''

    @tf.function
    def random_gamma(self, x):
        B = tf.shape(x)[0]
        gamma = tf.random.uniform([B,1,1,1,1], 0.7, 1.5)
        x = tf.pow(x, gamma)
        return tf.clip_by_value(x, 0., 1.)

    @tf.function
    def random_blur(self, x):
        #if tf.random.uniform([]) < 0.5:
        B = tf.shape(x)[0]
        blurred = tf.nn.avg_pool3d(
                x,
                ksize=3,
                strides=1,
                padding="SAME"
            )
        mask = tf.random.uniform([B,1,1,1,1]) < 0.5
        x = tf.where(mask, blurred, x)
        return x

    @tf.function
    def random_noise(self, x, std=0.05):
        return tf.clip_by_value(
            x + tf.random.normal(tf.shape(x), stddev=std),
            0., 1.
        )

    @tf.function
    def random_contrast(self, x):
        B = tf.shape(x)[0]
        factor = tf.random.uniform([B,1,1,1,1], 0.7, 1.3)
        mean = tf.reduce_mean(x, axis=[1,2,3,4], keepdims=True)
        x = (x - mean) * factor + mean
        return tf.clip_by_value(x, 0., 1.)

    @tf.function
    def strong_augment(self, x):
        x = self.random_gamma(x)
        # brightness
        #delta = tf.random.uniform([], -0.2, 0.2)
        #x = tf.clip_by_value(x + delta, 0., 1.)
        x = self.random_contrast(x)
        x = self.random_blur(x)
        x = self.random_noise(x, 0.05)
        return tf.clip_by_value(x, 0., 1.)

    @tf.function
    def update_teacher(self, decay=0.99):
        #decay = min(0.99, 1 - 1/(epoch+1))
        for w_t, w_s in zip(self.teacher.weights, self.student.weights):
            w_t.assign(decay * w_t + (1.0 - decay) * w_s)
            #w_t.assign(tf.cast(decay, w_t.dtype) * w_t +
            #    tf.cast(1.0 - decay, w_t.dtype) * w_s)

    @tf.function
    def train_step_pseudo(self, data):
        x_l = data["x_l"]
        y_l = data["y_l"]
        x_u = data["x_u"]

        with tf.GradientTape() as tape:

            # supervised
            y_pred_l = self.student(x_l, training=True)
            per_voxel_loss = self.ce(y_l, y_pred_l)
            per_example_loss = tf.reduce_mean(
                per_voxel_loss,
                axis=[1,2,3]  # reduce spatial dims only
            )
            loss_sup = tf.nn.compute_average_loss(
                per_example_loss,
                global_batch_size=self.batch_size
            )

            # --- UNSUPERVISED ---
            x_u = self.strong_augment(x_u)
            p_teacher = self.teacher(x_u, training=True)   # dropout perturbation
            pseudo = tf.stop_gradient(tf.cast(tf.argmax(p_teacher, axis=-1), tf.int32))
            conf = tf.reduce_max(p_teacher, axis=-1)
            weight = tf.clip_by_value((conf - 0.9) / 0.1, 0.0, 1.0)
            #weight = tf.sigmoid((conf - 0.9) * 10.0)

            p_student = self.student(x_u, training=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(
              pseudo, p_student, from_logits=False)
            per_example_unsup = tf.reduce_sum(weight * ce, axis=[1,2,3]) / (
              tf.reduce_sum(weight, axis=[1,2,3]) + 1e-6)

            loss_unsup = tf.nn.compute_average_loss(
                per_example_unsup,
                global_batch_size=self.batch_size #TODO: batch_size*4?
            )

            loss = loss_sup + self.ramp_weight(self.current_epoch) * loss_unsup

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.update_teacher()

        return {
            "loss": loss,
            "supervised": loss_sup,
            "unsupervised": loss_unsup,
        }

