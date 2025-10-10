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

### FEATURE ENGINEERING ###
#Making Recipe
target <- "action" 
ev <- c("role_title", "role_rollup_2", "role_rollup_1", "role_family_desc", 
        "role_family", "role_deptname", "role_code", "resource", "mgr_id")
amazon_recipe <- recipe(action ~ ., data = train) |>
  step_mutate_at(all_of(ev), fn = factor) |>
  step_other(all_of(ev), threshold = 0.001) |>
  step_dummy(all_of(ev))
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = test)

### WORK FLOWS ###
# (1) Defining Logistic Regression Model
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
  select(id, .pred_0) |>
  rename(action = .pred_0)

#Saving CSV File
vroom_write(x=kaggle_log, file="./Logistic.csv", delim=",")
