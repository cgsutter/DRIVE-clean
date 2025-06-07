# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    LayerNormalization,
    MaxPooling2D,
    Reshape,
)

import _config as config

# This whole script is one function

def model_baseline(
    # one_off,
    # hyp_run,
    evid,
    num_classes,
    input_shape,
    arch,
    transfer_learning,
    ast,
    dropout_rate,
    l2weight,
    activation_layer_def,
    activation_output_def,
):
    """
    Args:
        num_classes (int): number of output classes
        input_shape: size of input images, eg (224,224,3)
        arch (str): architecture. In main this is looped through from the list of architectures as defined in config
    """

    print("entering model build function")


    if evid:
        print(
            "evidential model with different output acivation, linear as set in config"
        )
        outputlayer_activation = evid_output_activation
    else:
        outputlayer_activation = activation_output_def

    # Define input shape and number of classes
    # input_shape = input_shape
    # num_classes = 1000  # Adjust based on your dataset

    if arch == "xcep":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.Xception(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                # if hyp_run == True:
                print(
                    "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
                )
                model.add(
                    Dense(
                        256,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.Xception(
                include_top=False,
                weights=None,
                input_shape=input_shape,  # [224,224,3]
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(GlobalAveragePooling2D())
            # if hyp_run == True:
            print(
                "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
            )
            model.add(
                Dense(
                    256,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation=outputlayer_activation))

    elif arch == "vgg16":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.VGG16(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(Flatten())
                model.add(
                    Dense(
                        units=4096,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        units=4096,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.VGG16(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print("Using architecture-specific top")
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(Flatten())
            model.add(
                Dense(
                    units=4096,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(
                Dense(
                    units=4096,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation=outputlayer_activation))

    elif arch == "resnet":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.ResNet50(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                # if hyp_run == True:
                print(
                    "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
                )
                model.add(
                    Dense(
                        256,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(GlobalAveragePooling2D())
            # if hyp_run == True:
            print(
                "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
            )
            model.add(
                Dense(
                    256,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation=outputlayer_activation))

    elif arch == "incep":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.InceptionV3(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                # if hyp_run == True:
                print(
                    "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
                )
                model.add(
                    Dense(
                        256,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(GlobalAveragePooling2D())
            # if hyp_run == True:
            print(
                "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
            )
            model.add(
                Dense(
                    256,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation=outputlayer_activation))

    elif arch == "mobilenet":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.MobileNet(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D(keepdims=True))
                # if hyp_run == True:
                print(
                    "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
                )
                model.add(
                    Dense(
                        256,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))

                model.add(
                    Conv2D(num_classes, (1, 1), padding="same", name="conv_preds")
                )
                model.add(Reshape((num_classes,), name="reshape_2"))

                model.add(
                    Activation(activation=outputlayer_activation, name="predictions")
                )
            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.MobileNet(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print("Using architecture-specific top")
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(GlobalAveragePooling2D(keepdims=True))
            # if hyp_run == True:
            print(
                "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
            )
            model.add(
                Dense(
                    256,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))

            model.add(Conv2D(num_classes, (1, 1), padding="same", name="conv_preds"))
            model.add(Reshape((num_classes,), name="reshape_2"))
            model.add(Activation(activation=outputlayer_activation, name="predictions"))
    elif arch == "densenet":
        print(f"building keras {arch}")

        if transfer_learning == True:
            print(
                "Using transfer learning with learned model weights from imagenet dataset for bottom layers. Retraining top dense layers."
            )

            # Load the pre-trained Xception model without the top layers (include_top=False)
            base_model = tf.keras.applications.DenseNet121(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze all layers in the base model
            base_model.trainable = False
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)

            # Add custom top layers for classification
            if ast == True:
                # Note: also used in non transfer learning as for that it's always architecture specific
                print("Using architecture-specific top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                # if hyp_run == True:
                print(
                    "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
                )
                model.add(
                    Dense(
                        256,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

            elif ast == False:
                # has dense layers already so don't need to add any specifically for hyp tuning
                print("Using generic top")
                print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
                model.add(GlobalAveragePooling2D())
                model.add(
                    Dense(
                        1024,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(
                    Dense(
                        512,
                        activation=activation_layer_def,
                        kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                    )
                )
                # if hyp_run == True:
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation=outputlayer_activation))

        elif transfer_learning == False:
            print("Training the full model, including the bottom (first) layers.")
            base_model = tf.keras.applications.DenseNet121(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation=outputlayer_activation,
            )
            # Freeze all layers in the base model
            base_model.trainable = True
            # Create a new sequential model and add the base model
            model = tf.keras.Sequential()
            model.add(base_model)
            # Note: this top is the same as in AST for trle above. Only ever ran one single top for
            print(f"l2 weight is {l2weight} and dropout rate is {dropout_rate}")
            model.add(GlobalAveragePooling2D())
            # if hyp_run == True:
            print(
                "Adjusted top layer with one added dense layer of size 256 and l2 regularization, followed by dropout layer -- as needed for hyperparameter tuning"
            )
            model.add(
                Dense(
                    256,
                    activation=activation_layer_def,
                    kernel_regularizer = None if l2weight == 0 else tf.keras.regularizers.l2(l=l2weight),
                )
            )
            model.add(Dropout(dropout_rate))
            model.add(Dense(num_classes, activation=outputlayer_activation))

    else:
        print("check architecture string input")

    # print(model.summary())

    return model
