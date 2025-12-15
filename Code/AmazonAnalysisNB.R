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
  step_lencode_glm(any_of(c("resource", "mgr_id")), outcome = target) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

### WORK FLOWS ###
# (5) Naive Bayes
library(naivebayes)
library(discrim)
#Defining Model
bayes_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_engine("naivebayes") |>
  set_mode("classification")

#Creating a Workflow
bayes_wf <- workflow() |>
  add_recipe(amazon_recipe)|>
  add_model(bayes_model)

#Defining Grid of Values
bayes_grid <- grid_regular(Laplace(range = c(0, 1)),
                           smoothness(range = c(0.001, 1)),
                           levels = 5) #3 for Testing; 5 for Results

#Splitting Data
bayes_folds <- vfold_cv(train,
                      v = 5,
                      repeats = 3) #1 for Testing; 3 for Results

#Run Cross Validation
bayes_results <- bayes_wf |>
  tune_grid(resamples = bayes_folds,
            grid = bayes_grid,
            metrics = metric_set(roc_auc))

#Find Best Tuning Parameters
bayes_best <- bayes_results |>
  select_best(metric = "roc_auc")

#Finalizing Workflow
final_bwf <- bayes_wf |>
  finalize_workflow(bayes_best) |>
  fit(data = train)

#Making Predictions
bayes_pred <- predict(final_bwf, new_data = test, type = "prob")

#Formatting Predictions for Kaggle
kaggle_bayes <- bayes_pred |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x = kaggle_bayes, file="./NB.csv", delim=",")

#End Parallel Computing
registerDoSEQ()
