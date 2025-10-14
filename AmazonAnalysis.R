### SET UP ###
#Downloading Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(janitor)
library(embed)
library(beepr)

#Bringing in Data
train <- vroom("train.csv") |> 
  clean_names() |>
  mutate(action = as.factor(action))
test <- vroom("test.csv") |>
  clean_names()

### FEATURE ENGINEERING ###
#Making Recipe
target <- "action" 
ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc", 
        "role_family", "role_deptname", "role_code", "resource", "mgr_id")
amazon_recipe <- recipe(action ~ ., data = train) |>
  step_mutate_at(any_of(ev), fn = factor) |>
  step_other(any_of(ev), threshold = 0.1) |> #0.1 for Testing; 0.001 for Results
  step_lencode_glm(any_of(ev), outcome = target) |>
  step_normalize(all_numeric_predictors())
prep <- prep(amazon_recipe)
beepr::beep()
baked <- bake(prep, new_data = test)

### WORK FLOWS ###
# (1) Logistic Regression
#Defining Model
logRegModel <- logistic_reg() |>
  set_engine("glm") 

#Making Workflow
log_workflow <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(logRegModel) |>
  fit(data = train)
beepr::beep()

#Making Predictions
amazon_predictions <- predict(log_workflow,
                              new_data=test,
                              type = "prob")
beepr::beep()

#Formatting Predictions for Kaggle
kaggle_log <- amazon_predictions |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x=kaggle_log, file="./Logistic.csv", delim=",")

# (2) Penalized Logistic Regression
#Defining Model
amazon_plog <- logistic_reg(mixture = tune(), penalty = tune()) |>
  set_engine("glmnet")

#Making Workflow
plog_workflow <- workflow() |>
  add_recipe(amazon_recipe) |>
  add_model(amazon_plog)

#Defining Grid of Values
tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 3) #3 for Testing; 5 for Results

#Splitting Data
folds <- vfold_cv(train, 
                  v = 5,
                  repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validation
CV_results <- plog_workflow |>
  tune_grid(resamples = folds, grid = tuning_grid, 
            metrics = metric_set(roc_auc)) 
beepr::beep()

#Find Best Tuning Parameters
bestTune <- CV_results |>
  select_best(metric = "roc_auc")

#Finalize Workflow
final_wf <- plog_workflow |>
  finalize_workflow(bestTune) |>
  fit(data = train)

#Making Predictions
plog_pred <- predict(final_wf, new_data = test, type = "prob")

#Formatting Predictions for Kaggle
kaggle_plog <- plog_pred |>
  bind_cols(test) |>
  select(id, .pred_1) |>
  rename(action = .pred_1)

#Saving CSV File
vroom_write(x=kaggle_plog, file="./Penalized2.csv", delim=",")

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