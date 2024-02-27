#!/usr/bin/env Rscript

## Train a simple MNIST classifier
## Export it for Tensorflow Serving

library(keras3)
use_backend("tensorflow")


## ---- Train the model ----

c(c(x_train, y_train), c(x_test, y_test)) %<-% (
  dataset_mnist() |> rapply(\(x) np_array(x, "uint8"), how = "replace")
)


input <- keras_input(shape = c(28, 28), dtype = "uint8")
output <- input |>
  layer_lambda(\(x) {
    x |>
      op_expand_dims(-1) |>
      op_cast("float32") |>
      op_divide(255)
  }) |>
  layer_conv_2d(32, c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(64, c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(c(2, 2)) |>
  layer_flatten() |>
  layer_dropout(rate = 0.5) |>
  layer_dense(10, activation = "softmax")

model <- keras_model(input, output, name = "mnist-classifier")
summary(model)


model |> compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

batch_size <- 128
epochs <- 15

model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.1
)

score <- model |> evaluate(x_test, y_test, verbose=0)
cat("Test set evaluation:\n"); str(score, no.list = TRUE)


## ---- Export the model for TensorFlow Serving ----
export_base_path <- "saved-models"
export_model_name <- "mnist-classifier"
export_model_version <- 1

export_path <- fs::path(export_base_path,
                        export_model_name,
                        export_model_version)

model |>
  export_savedmodel(export_path)
# Saved artifact at 'saved-models/mnist-classifier/1'. The following endpoints are available:
#
# * Endpoint 'serve'
#   args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 28, 28), dtype=tf.uint8, name='keras_tensor')
# Output Type:
#   TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
# Captures:
#   140387898530064: TensorSpec(shape=(), dtype=tf.resource, name=None)
#   140387898531024: TensorSpec(shape=(), dtype=tf.resource, name=None)
#   140387898530640: TensorSpec(shape=(), dtype=tf.resource, name=None)
#   140387898531792: TensorSpec(shape=(), dtype=tf.resource, name=None)
#   140387898530832: TensorSpec(shape=(), dtype=tf.resource, name=None)
#   140387898532944: TensorSpec(shape=(), dtype=tf.resource, name=None)


# confirm we can reload the model in R and serve it
reloaded_artifact <- tensorflow::tf$saved_model$load(export_path)
input_data <- x_train[0:1] # keep batch dim
predictions <- reloaded_artifact$serve(input_data)
stopifnot(predictions |> op_argmax() |> as.array() == 5)

## confirm the tensorflow serving cli recognizes the exported model artifact
system(sprintf("saved_model_cli show --dir '%s' --all", export_path))
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
#
# signature_def['__saved_model_init_op']:
#   The given SavedModel SignatureDef contains the following input(s):
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['__saved_model_init_op'] tensor_info:
#         dtype: DT_INVALID
#         shape: unknown_rank
#         name: NoOp
#   Method name is:
#
# signature_def['serve']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['keras_tensor'] tensor_info:
#         dtype: DT_UINT8
#         shape: (-1, 28, 28)
#         name: serve_keras_tensor:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['output_0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 10)
#         name: StatefulPartitionedCall:0
#   Method name is: tensorflow/serving/predict
#
# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['keras_tensor'] tensor_info:
#         dtype: DT_UINT8
#         shape: (-1, 28, 28)
#         name: serving_default_keras_tensor:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['output_0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 10)
#         name: StatefulPartitionedCall_1:0
#   Method name is: tensorflow/serving/predict
# The MetaGraph with tag set ['serve'] contains the following ops: {'VarIsInitializedOp', 'Identity', 'ReadVariableOp', 'SaveV2', 'NoOp', 'Pack', 'AddV2', 'StringJoin', 'Select', 'RealDiv', 'MatMul', 'MaxPool', 'Placeholder', 'Reshape', 'Const', 'AssignVariableOp', 'Relu', 'Cast', 'RestoreV2', 'MergeV2Checkpoints', 'Softmax', 'ExpandDims', 'StaticRegexFullMatch', 'VarHandleOp', 'ShardedFilename', 'DisableCopyOnRead', 'Conv2D', 'StatefulPartitionedCall'}
#
# Concrete Functions:
#   Function Name: 'serve'
#     Option #1
#       Callable with:
#         Argument #1
#           keras_tensor: TensorSpec(shape=(None, 28, 28), dtype=tf.uint8, name='keras_tensor')
