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

def consistency_loss(p1, p2):
    return ops.mean((p1 - p2) ** 2)

def kl_loss(p1, p2):
    eps = 1e-6
    conf = ops.max(p1, axis=-1, keepdims=True)
    # soft weighting instead of hard mask
    weight = ops.clip((conf - 0.9) / 0.1, 0.0, 1.0)
    kl = p1 * ops.log((p1 + eps) / (p2 + eps))
    return ops.sum(weight * kl) / (ops.sum(weight) + eps)

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
    def __init__(self, model, lambda_consistency=0.05, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.lambda_consistency = lambda_consistency
        self.ce = keras.losses.CategoricalCrossentropy()
        self.val_dice = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.current_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, x, training=False):
        if isinstance(x, dict):
            x = x["x_l"]
        return self.model(x, training=training)

    def ramp_weight(self, epoch, ramp_epochs=20):
        epoch = ops.cast(epoch, "float32")
        ramp_epochs = ops.cast(ramp_epochs, "float32")
        alpha = (epoch-20) / (ramp_epochs + 1e-6)
        alpha = ops.clip(alpha, 0.0, 1.0)
        return self.lambda_consistency * alpha

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = ops.mean(self.ce(y, y_pred))
        return {"val_loss": loss}

    def train_step(self, data):
        # unpack
        x_l = data["x_l"]
        y_l = data["y_l"]
        x_u = data["x_u"]

        with tf.GradientTape() as tape:

            # --- supervised branch ---
            y_pred_l = self(x_l, training=True)
            loss_sup = ops.mean(self.ce(y_l, y_pred_l))

            # --- unsupervised branch ---
            # x_u shape: (B, 9, 64, 64, 64, C)
            B = ops.shape(x_u)[0]
            num_patches = 9

            # flatten patches
            x_u_flat = ops.reshape(x_u, (B * num_patches, *x_u.shape[2:]))
            y_pred_u = self(x_u_flat, training=True)
            y_pred_u = ops.reshape(y_pred_u, (B, num_patches, *y_pred_u.shape[1:]))

            # consistency across patch pairs
            cons = 0.0
            count = 0
            p1 = y_pred_u[:, 0]
            for i, shift in enumerate([(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1),(-1,1,1),(1,-1,1),(1,1,-1),(1,1,1)]):
                p2 = y_pred_u[:, i+1]
                p2 = ops.stop_gradient(p2)
                o1, o2 = extract_overlap(p1, p2, shift)
                cons += kl_loss(o1, o2)
                #cons += consistency_loss(o1, o2)
                count += 1

            # unsupervised loss
            loss_unsup = cons / (count + 1e-6)

            # --- total ---
            #lambda_ssl = self.lambda_consistency * tf.sigmoid((self.val_dice - 0.7) * 20)
            loss = loss_sup + self.ramp_weight(self.current_epoch) * loss_unsup

        # gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": loss,
            "supervised": loss_sup,
            "unsupervised": loss_unsup,
        }

