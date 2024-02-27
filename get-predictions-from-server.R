

## assumes the server is up and running
# system("./run-model-server.sh &")

library(httr2)
library(glue)
library(purrr)

MODEL_NAME <- "mnist-classifier"
PORT <- 8501

# test that the server is up
req <- request(glue("http://localhost:{PORT}/v1/models/{MODEL_NAME}"))

req |> req_dry_run()
resp <- req |> req_perform()
resp
stopifnot(resp |>resp_status() == 200) # make sure server is up, ./run-model-server.sh

## To GET metadata about the serving model:
# curl http://localhost:8501/v1/models/mnist-classifier/metadata
# curl http://localhost:8501/v1/models/mnist-classifier/versions/1/metadata


## get predictions
input_data     <- keras3::dataset_mnist()$train$x[1:3,,]
expected_preds <- keras3::dataset_mnist()$train$y[1:3]

req <-
  request(glue(
    "http://localhost:{PORT}/v1/models/{MODEL_NAME}:predict"
  )) |>
  req_body_json(list(instances = as.array(input_data)),
                matrix = "rowmajor")

req |> req_dry_run()
resp <- req |> req_perform()
resp |> resp_raw()

preds <- resp |> resp_body_json()
# preds |> str()
preds <- preds$predictions |>
  map_int(\(probs) which.max(unlist(probs)) - 1)
stopifnot(preds == expected_preds)


## For models with multiple serving signatures,
## you can specify which "signature" as part of the POST data.
## e.g, this next example is the same as the previous one, except for
## "signature_name" in data, as seen in the "metadata"
req <-
  request(glue(
    "http://localhost:{PORT}/v1/models/{MODEL_NAME}:predict"
  )) |>
  req_body_json(list(signature_name = "serve",
                     instances = as.array(input_data)),
                matrix = "rowmajor")

# req |> req_dry_run()
resp <- req |> req_perform()
# resp <- tryCatch(req |> req_perform(), error = \(e) e$resp)
preds <- resp |> resp_body_json(simplifyVector = TRUE) |> _$predictions
stopifnot(expected_preds == (max.col(preds)-1))


