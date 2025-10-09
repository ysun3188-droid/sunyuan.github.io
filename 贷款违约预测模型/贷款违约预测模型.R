# Load necessary libraries
library(data.table)  # For data reading and manipulation
library(caret)       # For data partitioning and modeling
library(dplyr)       # For data manipulation
library(pROC)        # For AUC-ROC calculation

# Load the training and test datasets
train_data <- fread("application_train.csv")
test_data <- fread("application_test.csv")

# Step 1: Separate the TARGET column from the training data
y_train <- train_data$TARGET
train_data <- train_data %>% select(-TARGET)

# Step 2: Handle missing values by imputing the median for numeric columns
train_data <- train_data %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))

test_data <- test_data %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))

# Step 3: One-hot encoding for categorical variables
dummy_model_train <- dummyVars(~ ., data = train_data, fullRank = TRUE)
dummy_model_test <- dummyVars(~ ., data = test_data, fullRank = TRUE)

# Apply the dummy model to train and test datasets
train_data <- predict(dummy_model_train, newdata = train_data)
test_data <- predict(dummy_model_test, newdata = test_data)

# Convert the resulting matrices to data frames
train_data <- as.data.frame(train_data)
test_data <- as.data.frame(test_data)

# Ensure both datasets have the same columns
common_cols <- intersect(names(train_data), names(test_data))
train_data <- train_data[, common_cols]
test_data <- test_data[, common_cols]

# Step 4: Split the training data into training and validation sets
set.seed(42)
train_index <- createDataPartition(y_train, p = 0.8, list = FALSE)

X_train <- train_data[train_index, ]
X_val <- train_data[-train_index, ]
y_train_split <- y_train[train_index]
y_val <- y_train[-train_index]

# Step 5: Train the logistic regression model on the training data
log_model <- glm(y_train_split ~ ., data = X_train, family = binomial)

# Step 6: Predict on the validation set and evaluate using AUC
val_pred <- predict(log_model, newdata = X_val, type = "response")
roc_curve <- roc(y_val, val_pred)
auc_val <- auc(roc_curve)
print(paste("Validation AUC: ", auc_val))

# Step 7: Predict on the test set
test_pred <- predict(log_model, newdata = test_data, type = "response")

# Step 8: Ensure SK_ID_CURR is an integer in the test set
test_data$SK_ID_CURR <- as.integer(test_data$SK_ID_CURR)

# Step 9: Create the submission data frame with SK_ID_CURR and TARGET (predictions)
submission <- data.frame(SK_ID_CURR = test_data$SK_ID_CURR, TARGET = test_pred)

# Step 10: Turn off scientific notation for large numbers and write the CSV file
options(scipen = 999)

# Write the submission file in the required format
write.csv(submission, "submission.csv", row.names = FALSE, quote = FALSE)
