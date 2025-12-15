### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(janitor)
library(ranger)
library(doParallel)
library(embed)
library(themis)
# library(beepr)
# library(rstanarm)
# library(patchwork)
# library(kknn)
# library(discrim)
# library(naivebayes)
# library(keras)
# library(kernlab)

#Bringing in Data
train <- vroom("train.csv") |> 
  clean_names() |>
  mutate(action = as.factor(action))
test <- vroom("test.csv") |>
  clean_names()

#Setting Up Parallel Computing
registerDoParallel(cores = 5)

# ### FEATURE ENGINEERING ###
#Making Original Recipe
target <- "action"
ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
        "role_family", "role_deptname", "role_code", "resource", "mgr_id")
amazon_recipe <- recipe(action ~ ., data = train) |>
  step_mutate_at(any_of(ev), fn = factor) |>
  step_other(any_of(ev), threshold = 0.001) |> #0.1 for Testing; 0.001 for Results
  step_lencode_glm(any_of(ev), outcome = target) |>
  step_nzv(all_predictors())

# # (1) Making Principal Component Reduction Recipe
# target <- "action"
# ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
#         "role_family", "role_deptname", "role_code", "resource", "mgr_id")
# pcr_recipe <- recipe(action ~ ., data = train) |>
#   step_mutate_at(any_of(ev), fn = factor) |>
#   step_other(any_of(ev), threshold = 0.1) |> #0.1 for Testing; 0.001 for Results
#   step_dummy(any_of(ev)) |>
#   step_normalize(all_numeric_predictors()) |>
#   step_pca(all_predictors(), threshold = 0.95)
# prep <- prep(pcr_recipe)
# baked <- bake(prep, new_data = test)
# 
# # (2) Making SMOTE Recipe
# target <- "action"
# ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
#         "role_family", "role_deptname", "role_code", "resource", "mgr_id")
# smote_recipe <- recipe(formula = action ~ ., data = train) |>
#   step_mutate_at(any_of(ev), fn = factor) |>
#   step_other(any_of(ev), threshold = 0.001) |> #0.1 for Testing; 0.001 for Results
#   step_lencode_glm(any_of(ev), outcome = target) |>
#   step_normalize(all_numeric_predictors()) |>
#   step_smote(action)
# prep <- prep(smote_recipe)
# baked <- bake(prep, new_data = test)

# (3) Making a New Recipe
# target <- "action"
# ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
#         "role_family", "role_deptname", "role_code", "resource", "mgr_id")
# new_recipe <- recipe(action ~ ., data = train) |>
#   step_mutate_at(any_of(ev), fn = factor) |>
#   step_other(any_of(ev), threshold = 0.001) |> #0.1 for Testing; 0.001 for Results
#   step_lencode_bayes(any_of(ev), outcome = target)
# prep <- prep(new_recipe)
# baked <- bake(prep, new_data = test)
# 
### WORK FLOWS ###
# # (1) Logistic Regression
# #Defining Model
# logRegModel <- logistic_reg() |>
#   set_engine("glm")
# 
# #Making Workflow
# log_workflow <- workflow() |>
#   add_recipe(pcr_recipe) |>
#   add_model(logRegModel) |>
#   fit(data = train)
# 
# #Making Predictions
# amazon_predictions <- predict(log_workflow,
#                               new_data=test,
#                               type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_log <- amazon_predictions |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x=kaggle_log, file="./Logistic(2).csv", delim=",")
# 
# # (2) Penalized Logistic Regression
# #Defining Model
# amazon_plog <- logistic_reg(mixture = tune(), penalty = tune()) |>
#   set_engine("glmnet")
# 
# #Making Workflow
# plog_workflow <- workflow() |>
#   add_recipe(pcr_recipe) |>
#   add_model(amazon_plog)
# 
# #Defining Grid of Values
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 3) #3 for Testing; 5 for Results
# 
# #Splitting Data
# folds <- vfold_cv(train,
#                   v = 5,
#                   repeats = 1) #1 for Testing; 3 for Results
# 
# #Run Cross Validation
# CV_results <- plog_workflow |>
#   tune_grid(resamples = folds, grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# bestTune <- CV_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalize Workflow
# final_wf <- plog_workflow |>
#   finalize_workflow(bestTune) |>
#   fit(data = train)
# 
# #Making Predictions
# plog_pred <- predict(final_wf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_plog <- plog_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x=kaggle_plog, file="./Penalized_PCR.csv", delim=",")

# (3) Random Forests
#Defining Model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 500) |> #50 for Testing; 1000 for Results
  set_engine("ranger") |>
  set_mode("classification")

#Creating a Workflow
forest_wf <- workflow() |>
  add_recipe(amazon_recipe)|>
  add_model(forest_model)

#Defining Grid of Values
max <- ncol(baked)
forest_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = 3) #3 for Testing; 5 for Results

#Splitting Data
forest_folds <- vfold_cv(train,
                         v = 10,
                         repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validation
forest_results <- forest_wf |>
  tune_grid(resamples = forest_folds,
            grid = forest_grid,
            metrics = metric_set(roc_auc))

#Find Best Tuning Parameters
bestTune <- forest_results |>
  select_best(metric = "roc_auc")

#Finalizing Workflow with Cross Validation
final_fwf <- forest_wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)

#Making Predictions
forest_pred <- predict(final_fwf, new_data = test, type = "prob")

#Formatting Predictions for Kaggle
kaggle_forest <- forest_pred |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x=kaggle_forest, file="./RF2.csv", delim=",")

#Saving Results
saveRDS(final_fwf, "forest_model2.rds")

# #Bringing in Best Results
# bestTune <- readRDS("forest_model.rds")
# 
# # Recreate the workflow
# forest_wf <- workflow() |>
#   add_recipe(amazon_recipe) |>
#   add_model(
#     rand_forest(
#       mtry = 3,
#       min_n = 2,
#       trees = 2000
#     ) |> 
#       set_engine("ranger") |>
#       set_mode("classification")
#   )
# 
# # Fit on the full training data
# final_fwf <- fit(forest_wf, data = train)
# 
# #Making Predictions
# forest_pred <- predict(final_fwf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_forest <- forest_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x=kaggle_forest, file="./Forest_BATCH2.csv", delim=",")

# # (4) K-Nearest Neighbors
# #Defining Model
# knn_model <- nearest_neighbor(neighbors = tune()) |>
#   set_engine("kknn") |>
#   set_mode("classification")
# 
# #Creating a Workflow
# knn_wf <- workflow() |>
#   add_recipe(pcr_recipe)|>
#   add_model(knn_model)
# 
# #Defining Grid of Values
# knn_grid <- grid_regular(neighbors(range = c(1, 250)),
#                             levels = 5) #More Levels = More Time
# 
# #Splitting Data
# knn_folds <- vfold_cv(train,
#                          v = 5,
#                          repeats = 1) #1 for Testing; 3 for Results
# 
# #Run Cross Validation
# knn_results <- knn_wf |>
#   tune_grid(resamples = knn_folds,
#             grid = knn_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# knn_best <- knn_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalizing Workflow
# final_kwf <- knn_wf |>
#   finalize_workflow(knn_best) |>
#   fit(data = train)
# 
# #Making Predictions
# knn_pred <- predict(final_kwf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_knn <- knn_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x=kaggle_knn, file="./KNN_PCR.csv", delim=",")
# 
# # (5) Naive Bayes
# #Defining Model
# bayes_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
#   set_engine("naivebayes") |>
#   set_mode("classification")
# 
# #Creating a Workflow
# bayes_wf <- workflow() |>
#   add_recipe(amazon_recipe)|>
#   add_model(bayes_model)
# 
# #Defining Grid of Values
# bayes_grid <- grid_regular(Laplace(range = c(0, 2)),
#                            smoothness(range = c(0.01, 1)),
#                            levels = 5) #3 for Testing; 5 for Results
# 
# #Splitting Data
# bayes_folds <- vfold_cv(train,
#                       v = 5,
#                       repeats = 3) #1 for Testing; 3 for Results
# 
# #Run Cross Validation
# bayes_results <- bayes_wf |>
#   tune_grid(resamples = bayes_folds,
#             grid = bayes_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# bayes_best <- bayes_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalizing Workflow
# final_bwf <- bayes_wf |>
#   finalize_workflow(bayes_best) |>
#   fit(data = train)
# 
# #Making Predictions
# bayes_pred <- predict(final_bwf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_bayes <- bayes_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x = kaggle_bayes, file="./Bayes.csv", delim=",")
# 
# # (6) Neural Networks
# #Making Recipe
# target <- "action"
# ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc",
#         "role_family", "role_deptname", "role_code", "resource", "mgr_id")
# nn_recipe <- recipe(formula = action ~ ., data = train) |>
#   step_mutate_at(any_of(ev), fn = factor) |>
#   step_other(any_of(ev), threshold = 0.1) |> #0.1 for Testing; 0.001 for Results
#   step_lencode_glm(any_of(ev), outcome = target) |>
#   step_normalize(all_numeric_predictors())
# 
# #Defining Model
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50) |> #50 Low, 100 Medium, 250 High
#   set_engine("keras") |>
#   set_mode("classification")
# 
# #Creating a Workflow
# nn_wf <- workflow() |>
#   add_recipe(nn_recipe)|>
#   add_model(nn_model)
# 
# #Defining Grid of Values
# nn_grid <- grid_regular(hidden_units(range = c(1, 20)),
#                            levels = 3) #3 for Testing; 5 for Results
# 
# #Splitting Data
# nn_folds <- vfold_cv(train,
#                       v = 5,
#                       repeats = 1) #1 for Testing; 3 for Results
# 
# #Run Cross Validation
# nn_results <- nn_wf |>
#   tune_grid(resamples = nn_folds,
#             grid = nn_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# nn_best <- nn_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalizing Workflow
# final_nwf <- nn_wf |>
#   finalize_workflow(nn_best) |>
#   fit(data = train)
# 
# #Making Predictions
# nn_pred <- predict(final_nwf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_nn <- nn_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x = kaggle_nn, file="./NN_BATCH.csv", delim=",")
# 
# #Making Plot
# nn_results |> collect_metrics() |>
#   filter(.metric == "roc_auc") |>
#   ggplot(aes(x = hidden_units, y = mean)) +
#   geom_line() +
#   geom_point() +
#   labs(title = "ROC AUC Vs Hidden Units",
#        x = "Hidden Units",
#        y = "Mean ROC AUC")
# 
# # (7) Support Vector Machines
# #Defining Linear Model
# svml_model <- svm_linear(cost = tune()) |> 
#   set_engine("kernlab") |>
#   set_mode("classification")
# 
# #Defining Polynomial Model
# svmp_model <- svm_poly(degree = tune(), cost = tune()) |> 
#   set_engine("kernlab") |>
#   set_mode("classification")
# 
# #Defining Radial Model
# svmr_model <- svm_rbf(rbf_sigma = tune(), cost = tune()) |> 
#   set_engine("kernlab") |>
#   set_mode("classification")
# 
# #Creating Workflows
# #Linear
# svml_wf <- workflow() |>
#   add_recipe(pcr_recipe)|>
#   add_model(svml_model)
# #Polynomial
# svmp_wf <- workflow() |>
#   add_recipe(pcr_recipe)|>
#   add_model(svmp_model)
# #Radial
# svmr_wf <- workflow() |>
#   add_recipe(pcr_recipe)|>
#   add_model(svmr_model)
# 
# #Defining Grids of Values
# svml_grid <- grid_regular(cost(range = c(-2,2)),
#                         levels = 3) #1 for Testing; 3 for Results
# svmp_grid <- grid_regular(cost(range = c(-2,2)),
#                           degree(range = c(2,5)),
#                           levels = 3) #1 for Testing; 3 for Results
# svmr_grid <- grid_regular(cost(range = c(-2,2)),
#                           rbf_sigma(range = c(-1,1)),
#                           levels = 3) #1 for Testing; 3 for Results
# 
# #Splitting Data
# svml_folds <- vfold_cv(train,
#                      v = 3, #2 for Testing; 3 for Results
#                      repeats = 3) #1 for Testing; 3 for Results
# svmp_folds <- vfold_cv(train,
#                        v = 3, #2 for Testing; 3 for Results
#                        repeats = 3) #1 for Testing; 3 for Results
# svmr_folds <- vfold_cv(train,
#                        v = 3, #2 for Testing; 3 for Results
#                        repeats = 3) #1 for Testing; 3 for Results
# 
# #Run Cross Validations
# svml_results <- svml_wf |>
#   tune_grid(resamples = svml_folds,
#             grid = svml_grid,
#             metrics = metric_set(roc_auc))
# svmp_results <- svmp_wf |>
#   tune_grid(resamples = svmp_folds,
#             grid = svmp_grid,
#             metrics = metric_set(roc_auc))
# svmr_results <- svmr_wf |>
#   tune_grid(resamples = svmr_folds,
#             grid = svmr_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# svml_best <- svml_results |>
#   select_best(metric = "roc_auc")
# svmp_best <- svmp_results |>
#   select_best(metric = "roc_auc")
# svmr_best <- svmr_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalizing Workflow
# final_svml_wf <- svml_wf |>
#   finalize_workflow(svml_best) |>
#   fit(data = train)
# final_svmp_wf <- svmp_wf |>
#   finalize_workflow(svmp_best) |>
#   fit(data = train)
# final_svmr_wf <- svmr_wf |>
#   finalize_workflow(svmr_best) |>
#   fit(data = train)
# 
# #Making Predictions
# svml_pred <- predict(final_svml_wf, new_data = test, type = "prob")
# svmp_pred <- predict(final_svmp_wf, new_data = test, type = "prob")
# svmr_pred <- predict(final_svmr_wf, new_data = test, type = "prob")
# 
# #Formatting Predictions for Kaggle
# kaggle_svml <- svml_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# kaggle_svmp <- svmp_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# kaggle_svmr <- svmr_pred |>
#   bind_cols(test) |>
#   select(id, .pred_1) |>
#   rename(action = .pred_1)
# 
# #Saving CSV File
# vroom_write(x = kaggle_svml, file="./SVML_BATCH.csv", delim=",")
# vroom_write(x = kaggle_svmp, file="./SVMP_BATCH.csv", delim=",")
# vroom_write(x = kaggle_svmr, file="./SVMR_BATCH.csv", delim=",")

# ### EDA ### 
# #Wrangling Data for EDA
# train_long <- train |>
#   pivot_longer(
#     cols = c("resource", "mgr_id", "role_rollup_1", "role_rollup_2", "role_code",
#              "role_deptname", "role_title", "role_family_desc","role_family"),
#     names_to = "variable",
#     values_to = "value"
#   )
# 
# #Boxplot of Explanatory Variables
# plot1 <- ggplot(data = train_long, aes(x = value, y = variable, 
#                                        fill = variable)) +
#   geom_boxplot() +
#   scale_fill_brewer() + 
#   labs(
#     x = "",
#     y = "",
#     title = "Distribution of EV"
#   )
# plot1
# 
# #Bar Chart of Action
# plot2 <- ggplot(train, aes(x = factor(action))) +  
#   geom_bar(fill = "dodgerblue") +
#   coord_flip() +
#   labs(
#     x = "Action",
#     y = "Count",
#     title = "Distribution of Action"
#   ) 
# plot2
# 
# #Putting Plots Together
# plot1 / plot2

#End Parallel Computing
registerDoSEQ()
