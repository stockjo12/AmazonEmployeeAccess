library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(themis)
library(beepr)

# Read in test and train data sets
am_train <- vroom("train.csv")
am_test <- vroom("test.csv")

am_train$ACTION <- as.factor(am_train$ACTION)

# RECIPE
am_recipe <- recipe(ACTION ~ ., data = am_train) |>
  step_mutate_at(all_numeric_predictors(), fn = factor)|>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())


# Random Forests
rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(rf_mod)

folds_rf <- vfold_cv(am_train, v = 10, repeats = 1)

rf_grid <- grid_regular(mtry(range = c(1, 9)),
                        min_n(),
                        levels = 3)

CV_results_rf <- rf_wf |>
  tune_grid(resamples = folds_rf,
            grid = rf_grid,
            metrics = metric_set(roc_auc)) 
beepr::beep()

rf_best <- CV_results_rf |>
  select_best(metric = "roc_auc")

rf_final <- finalize_workflow(rf_wf, rf_best) |>
  fit(data = am_train)

rf_predictions <- rf_final |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(rf_predictions, "./RFPreds3.csv", delim = ",")

rf_mod_final <- rand_forest(
  mtry  = rf_best$mtry,
  min_n = rf_best$min_n,
  trees = 1500   # or 800–1500, whatever your machine tolerates
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_final <- workflow() |>
  add_recipe(am_recipe) |>   # your good dummy recipe
  add_model(rf_mod_final) |>
  fit(data = am_train)
