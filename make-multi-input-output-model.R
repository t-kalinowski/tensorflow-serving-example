library(keras3)
Sys.setenv("CUDA_VISIBLE_DEVICES"="")

num_tags <- 12  # Number of unique issue tags
num_words <- 10000  # Size of vocabulary obtained when preprocessing text data
num_departments <- 4  # Number of departments for predictions

title_input <- # Variable-length sequence of ints
  keras_input(shape(NA), name = "title")
body_input <-  # Variable-length sequence of ints
  keras_input(shape(NA), name = "body")
tags_input <-  # Binary vectors of size `num_tags`
  keras_input(shape = num_tags, name = "tags")

# Embed each word in the title into a 64-dimensional vector
title_features <- layer_embedding(title_input, num_words, 64)
# Embed each word in the text into a 64-dimensional vector
body_features <- layer_embedding(body_input, num_words, 64)

# Reduce sequence of embedded words in the title
# into a single 128-dimensional vector
title_features <- layer_lstm(title_features, 128)
# Reduce sequence of embedded words in the body
# into a single 32-dimensional vector
body_features <- layer_lstm(body_features, 32)

# Merge all available features into a single large vector via concatenation
x <- layer_concatenate(title_features, body_features, tags_input)

# Stick a logistic regression for priority prediction on top of the features
priority_pred <- layer_dense(x, 1, name = "priority")

# Stick a department classifier on top of the features
department_pred <- layer_dense(x, num_departments, name = "department")

# Instantiate an end-to-end model predicting both priority and department
model <- keras_model(
  inputs = list(title_input, body_input, tags_input),
  outputs = list(priority = priority_pred, department = department_pred)
)

model |> compile(
  optimizer = optimizer_rmsprop(1e-3),
  loss = list(
    priority = loss_binary_crossentropy(from_logits = TRUE),
    department = loss_categorical_crossentropy(from_logits = TRUE)
  ),
  loss_weights = c(priority = 1.0, department = 0.2)
)


# Dummy input data
batch_size <- 32
n_cases <- batch_size * 2
title_data <- random_integer(c(n_cases, 5), 0, num_words)
body_data <- random_integer(c(n_cases, 10), 0, num_words)
tags_data <- random_integer(c(n_cases, num_tags), 0, 2)

# Dummy target data
priority_targets <- random_normal(c(n_cases, 1))
dept_targets <- random_integer(c(n_cases, num_departments), 0, 2)

model |> fit(
  list(title = title_data, body = body_data, tags = tags_data),
  list(priority = priority_targets, department = dept_targets),
  epochs = 1,
  batch_size = batch_size
)

export_base_path <- "saved-models"
export_model_name <- "customer-ticket-assessor"
export_model_version <- 1

export_path <- fs::path(export_base_path,
                        export_model_name,
                        export_model_version)


# default saved model path
model |>
  export_savedmodel(export_path)


MODEL_NAME <- export_model_name
PORT <- 8501

drop <- TRUE
library(httr2)
library(glue)


req <-
  request(glue("http://localhost:{PORT}/v1/models/{MODEL_NAME}:predict")) |>
  req_body_json(list(inputs = list(
    title = as.array(title_data[1:3, , drop = FALSE]),
    body = as.array(body_data[1:3, , drop = FALSE]),
    tags = as.array(tags_data[1:3, , drop = FALSE])
  )),
  matrix = "rowmajor",
  pretty = TRUE)
req |> req_dry_run()
resp <- req |> req_perform()
last_response()
last_response() |> resp_body_json(simplifyVector = TRUE)
last_response() |> resp_body_json() |> _$error |> cat()
resp |> resp_raw()


# build up a custom saved model with additional signatures

library(tensorflow)
as_spec <- function(tensor, name = tensor$name) tf$TensorSpec(tensor$shape, dtype=tensor$dtype, name=name)
tf$nest$map_structure(as_spec, model$inputs)

export_archive <- keras$export$ExportArchive()
export_archive$track(model)
# first added sig is the "serving_default"
export_archive$add_endpoint(
  name = "serve",
  fn = tf$autograph$experimental$do_not_convert(\(...) {
    str(list(...))
    model(...)
  }),
  input_signature = list(list(
    as_spec(title_input),
    as_spec(body_input),
    as_spec(tags_input)
  ))
)

#tf$autograph$experimental$do_not_convert
export_archive$add_endpoint(
  name = "serve_unnameed",
  fn = (\(...) {
    str(list(...))
    x <- list(...)
    model(x)
  }),
  input_signature = list(as_spec(title_input, NULL),
                         as_spec(body_input, NULL),
                         as_spec(tags_input, NULL))
  )

export_archive$write_out(export_path)

#
# export_archive$add_endpoint(
#   name = "serve_nested_list",
#   fn = \(args) {
#     browser()
#     model(!!!args)
#   },
#   input_signature = list(list(
#     as_spec(title_input),
#     as_spec(body_input),
#     as_spec(tags_input)
#   )))
#
# export_archive$add_endpoint(
#   name = "serve_priority",
#   fn = \(...) model(...)$priority,
#   input_signature = list(
#     as_spec(title_input),
#     as_spec(body_input),
#     as_spec(tags_input)
#   )
# )

# export_archive$add_endpoint(
#   name = "serve_department",
#   fn = \(...) model(...)$department,
#   input_signature = list(
#     as_spec(title_input),
#     as_spec(body_input),
#     as_spec(tags_input)
#   )
# )



