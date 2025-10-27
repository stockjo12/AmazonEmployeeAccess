### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(janitor)
library(embed)
library(beepr)
library(ranger)
library(kknn) 
library(doParallel)
library(discrim)
library(naivebayes)
library(keras)

#Bringing in Data
train <- vroom("train.csv") |> 
  clean_names() |>
  mutate(action = as.factor(action))
test <- vroom("test.csv") |>
  clean_names()

#Setting Up Parallel Computing
registerDoParallel(cores = 5)

# ### FEATURE ENGINEERING ###
# #Making Recipe
# target <- "action" 
# ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc", 
#         "role_family", "role_deptname", "role_code", "resource", "mgr_id")
# amazon_recipe <- recipe(action ~ ., data = train) |>
#   step_mutate_at(any_of(ev), fn = factor) |>
#   step_other(any_of(ev), threshold = 0.001) |> #0.1 for Testing; 0.001 for Results
#   step_lencode_glm(any_of(ev), outcome = target) |>
#   step_normalize(all_numeric_predictors())
# prep <- prep(amazon_recipe)
# baked <- bake(prep, new_data = test)
# 
# ### WORK FLOWS ###
# # (1) Logistic Regression
# #Defining Model
# logRegModel <- logistic_reg() |>
#   set_engine("glm")
# 
# #Making Workflow
# log_workflow <- workflow() |>
#   add_recipe(amazon_recipe) |>
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
# vroom_write(x=kaggle_log, file="./Logistic_BATCH.csv", delim=",")
# 
# # (2) Penalized Logistic Regression
# #Defining Model
# amazon_plog <- logistic_reg(mixture = tune(), penalty = tune()) |>
#   set_engine("glmnet")
# 
# #Making Workflow
# plog_workflow <- workflow() |>
#   add_recipe(amazon_recipe) |>
#   add_model(amazon_plog)
# 
# #Defining Grid of Values
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) #3 for Testing; 5 for Results
# 
# #Splitting Data
# folds <- vfold_cv(train,
#                   v = 5,
#                   repeats = 3) #1 for Testing; 3 for Results
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
# vroom_write(x=kaggle_plog, file="./Penalized_BATCH.csv", delim=",")
# 
# # (3) Random Forests
# #Defining Model
# forest_model <- rand_forest(mtry = tune(),
#                             min_n = tune(),
#                             trees = 1000) |>
#   set_engine("ranger") |>
#   set_mode("classification")
# 
# #Creating a Workflow
# forest_wf <- workflow() |>
#   add_recipe(amazon_recipe)|>
#   add_model(forest_model)
# 
# #Defining Grid of Values
# maxNumXs <- ncol(baked)
# forest_grid <- grid_regular(mtry(range = c(1, maxNumXs)),
#                             min_n(),
#                             levels = 5) #3 for Testing; 5 for Results
# 
# #Splitting Data
# forest_folds <- vfold_cv(train,
#                          v = 5,
#                          repeats = 3) #1 for Testing; 3 for Results
# 
# #Run Cross Validation
# forest_results <- forest_wf |>
#   tune_grid(resamples = forest_folds,
#             grid = forest_grid,
#             metrics = metric_set(roc_auc))
# 
# #Find Best Tuning Parameters
# bestTune <- forest_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalizing Workflow
# final_fwf <- forest_wf |>
#   finalize_workflow(bestTune) |>
#   fit(data = train)
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
# vroom_write(x=kaggle_forest, file="./Forest_BATCH.csv", delim=",")
# 
# # (4) K-Nearest Neighbors
# #Defining Model
# knn_model <- nearest_neighbor(neighbors = tune()) |>
#   set_engine("kknn") |>
#   set_mode("classification")
# 
# #Creating a Workflow
# knn_wf <- workflow() |>
#   add_recipe(amazon_recipe)|>
#   add_model(knn_model)
# 
# #Defining Grid of Values
# knn_grid <- grid_regular(neighbors(range = c(1, 250)),
#                             levels = 25) #More Levels = More Time
# 
# #Splitting Data
# knn_folds <- vfold_cv(train,
#                          v = 5,
#                          repeats = 3) #1 for Testing; 3 for Results
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
# vroom_write(x=kaggle_knn, file="./KNN.csv", delim=",")
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

# (6) Neural Networks
#Making Recipe
target <- "action" 
ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc", 
        "role_family", "role_deptname", "role_code", "resource", "mgr_id")
nn_recipe <- recipe(formula = action ~ ., data = train) |>
  step_mutate_at(any_of(ev), fn = factor) |>
  step_other(any_of(ev), threshold = 0.1) |> #0.1 for Testing; 0.001 for Results
  step_lencode_glm(any_of(ev), outcome = target) |>
  step_normalize(all_numeric_predictors()) 

#Defining Model
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) |> #50 Low, 100 Medium, 250 High
  set_engine("keras") |>
  set_mode("classification")

#Creating a Workflow
nn_wf <- workflow() |>
  add_recipe(nn_recipe)|>
  add_model(nn_model)

#Defining Grid of Values
nn_grid <- grid_regular(hidden_units(range = c(1, 20)),
                           levels = 3) #3 for Testing; 5 for Results

#Splitting Data
nn_folds <- vfold_cv(train,
                      v = 5,
                      repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validation
nn_results <- nn_wf |>
  tune_grid(resamples = nn_folds,
            grid = nn_grid,
            metrics = metric_set(roc_auc))

#Find Best Tuning Parameters
nn_best <- nn_results |>
  select_best(metric = "roc_auc")

#Finalizing Workflow
final_nwf <- nn_wf |>
  finalize_workflow(nn_best) |>
  fit(data = train)

#Making Predictions
nn_pred <- predict(final_nwf, new_data = test, type = "prob")

#Formatting Predictions for Kaggle
kaggle_nn <- nn_pred |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x = kaggle_nn, file="./NN_BATCH.csv", delim=",")

#Making Plot
nn_results |> collect_metrics() |>
  filter(.metric == "roc_auc") |>
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() + 
  geom_point() +
  labs(title = "ROC AUC Vs Hidden Units",
       x = "Hidden Units",
       y = "Mean ROC AUC")

### EDA ### 
#Wrangling Data for EDA
train_long <- train |>
  pivot_longer(
    cols = c("resource", "mgr_id", "role_rollup_1", "role_rollup_2", "role_code",
             "role_deptname", "role_title", "role_family_desc","role_family"),
    names_to = "variable",
    values_to = "value"
  )

#Boxplot of Explanatory Variables
plot1 <- ggplot(data = train_long, aes(x = value, y = variable, 
                                       fill = variable)) +
  geom_boxplot() +
  scale_fill_brewer() + 
  labs(
    x = "",
    y = "",
    title = "Distribution of EV"
  )
plot1

#Bar Chart of Action
plot2 <- ggplot(train, aes(x = factor(action))) +  
  geom_bar(fill = "dodgerblue") +
  coord_flip() +
  labs(
    x = "Action",
    y = "Count",
    title = "Distribution of Action"
  ) 
plot2

#Putting Plots Together
plot1 / plot2

#End Parallel Computing
registerDoSEQ()