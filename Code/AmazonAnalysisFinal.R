### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(beepr)

#Bringing in Data
library(vroom)
library(janitor)
train <- vroom("train.csv") |> 
  clean_names() |>
  mutate(action = as.factor(action))
test <- vroom("test.csv") |>
  clean_names()

#Setting Up Parallel Computing
library(doParallel)
registerDoParallel(cores = 5)

### FEATURE ENGINEERING ###
library(embed)
library(themis)
#Making Original Recipe
target <- "action"
ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
        "role_family", "role_deptname", "role_code", "resource", "mgr_id")
amazon_recipe <- recipe(action ~ ., data = train) |>
  step_mutate(across(any_of(ev), as.factor)) |>
  step_other(any_of(ev), threshold = 0.001) |> #0.001 Threshold
  step_lencode_mixed(any_of(ev), outcome = target) |>
  step_upsample(action, over_ratio = 0.5) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

### WORK FLOWS ###
# (3) Random Forests
library(ranger)
library(dials)
#Defining Model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 1000) |> #500 Trees
  set_engine("ranger") |>
  set_mode("classification")

#Creating a Workflow
forest_wf <- workflow() |>
  add_recipe(amazon_recipe)|>
  add_model(forest_model)

#Defining Grid of Values
forest_grid <- grid_regular(mtry(range = c(1,9)), #Range c(1,9)
                            min_n(),
                            levels = 1) #3 Levels

#Splitting Data
forest_folds <- vfold_cv(train,
                         v = 10, #10 Folds
                         repeats = 3, #1 Repeat
                         strata = action) 

#Run Cross Validation
forest_results <- forest_wf |>
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(roc_auc))
beepr::beep()

#Find Best Tuning Parameters
bestTune <- forest_results |>
  select_best(metric = "roc_auc")

#Finalizing Workflow with Cross Validation
final_fwf <- forest_wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)

#Making Predictions
forest_pred <- predict(final_fwf, new_data = test, type = "prob")
beepr::beep()

#Formatting Predictions for Kaggle
kaggle_forest <- forest_pred |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x=kaggle_forest, file="./RF9_TEST17.csv", delim=",")

#End Parallel Computing
registerDoSEQ()
