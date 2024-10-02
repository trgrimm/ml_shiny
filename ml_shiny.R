library(MASS)
library(shiny)
library(bslib)
library(tidyverse)
library(tidymodels)
library(xgboost)
library(randomForest)
library(kknn)
library(palmerpenguins)
library(DT)

library(doFuture)

# Define UI for the application
ui <- navbarPage(
  title = "Shiny App for Machine Learning",
  theme = bs_theme(version = 5, preset = 'cosmo'),
  tabPanel("App Info",
           HTML('<h3>Explanation of App Usage</h3>
                <div>This app can be used to compare and evaluate the performance of various supervised machine learning (ML) methods on a dataset
                of choice. Once a model has been selected, the model can be fully trained and used to make predictions on new data.</div>
                <br>
                <div>The app contains multiple pages and is intended to be used in the following order: <b>Compare Models</b>, <b>Single Model Evaluation</b>, and <b>Prediction</b>.
                General information for each page is given below.</div>
                <br>
                <h4>Compare Models</h4>
                 <ul>
                  <li>Quickly compare the performance of multiple ML models using <b>default parameters</b>.</li>
                  <li>View the cross-validation (CV) performance across several metrics for each method.</li>
                  <li>Plot the  performance of each method for selected metrics.</li>
                </ul> 
                <h4>Single Model Evaluation</h4>
                <ul>
                  <li>Fit a single model using <b>default parameters</b> or perform <b>automatic</b> model tuning.</li>
                  <ul>
                  <li>If <b>automatic</b> tuning is selected, the model is tuned to optimize a selected performance metric.</li>
                  </ul>
                  <li>View the CV performance across various performance metrics using 80% of the data.</li>
                  <li>Plot the performance of a method on a 20% testing set.</li>
                </ul> 
                <h4>Prediction</h4>
                <ul>
                  <li>Fit a model to the entire dataset provided.</li>
                  <li>Use the fitted model to obtain predictions for new data.</li>
                </ul>
                <br>
                <h6><b>Important Note:</b> The option to "Enable parallelization?" on the <b>Compare Models</b> and <b>Single Model Evaluation</b> pages
                does not work on the version of the app that is currently hosted online. The app is currently hosted using a free account on 
                 <a href="https://www.shinyapps.io/">shinyapps.io</a>, which only uses a single core for computing.</h6>')),
  tabPanel("Compare Models",
           sidebarLayout(
             sidebarPanel(
               width = 3,
               selectInput("dataset_choice", "Select Dataset", choices = c("Palmer Penguins", "Boston Housing", "Upload Dataset"),
                           selected = 'Palmer Penguins'),
               
               # Conditional file input, visible only if 'Upload Dataset' is selected
               conditionalPanel(
                 condition = "input.compare_dataset_choice == 'Upload Dataset'",
                 fileInput("dataset", "Upload .csv File", accept = c(".csv"))
               ),
               uiOutput('compare_target'),
               uiOutput("compare_select_columns"),
               uiOutput("compare_select_categorical"),
               uiOutput('compare_ml_type'),
               uiOutput('compare_model'),
               checkboxGroupInput("preproc", 'Data Preprocessing',
                                  c("Yeo-Johnson Transformation" = 'yeo_johnson',
                                    "Center" = 'center',
                                    "Scale" = 'scale',
                                    "Impute NA's" = 'impute'
                                  )),
               conditionalPanel(
                 condition = "$.inArray('impute', input.preproc) > -1",
                 radioButtons("impute_type", 'Imputation Method',
                              c("Median" = 'med',
                                "Mean" = 'mean',
                                "KNN" = 'knn',
                                "Linear" = 'linear'
                              ))
               ),
               conditionalPanel(
                 condition = "$.inArray('impute', input.preproc) > -1",
                 radioButtons("cat_impute", "Impute Categorical NA's?",
                              c("Yes", "No"), selected = 'No')
               ),
               checkboxInput('parallel', label = "Enable Parallelization?", value = FALSE),
               conditionalPanel(
                 condition = "input.parallel == true",
                 numericInput('num_cores', label = 'Number of Cores', value = 2, min = 2, max = parallel::detectCores())
               ),
               actionButton("compare_train", "Train Model")
               
             ),
             mainPanel(
               width = 9,
               htmlOutput('model_selected_text'),
               
               navset_card_underline(
                 nav_panel('Data', DTOutput("compare_data_table"),
                           p(''),
                           uiOutput('compare_data_rows'),
                           br(),
                           uiOutput('compare_data_cols')),
                 nav_panel('Model Info',
                           p('Note the following:'),
                           HTML('<ul><li>10-fold cross-validation (CV) is used to evaluate model performance.</li><li>Default parameters are used for each model.</li></ul>'),
                           uiOutput('compare_model_info_variables')#,
                           # h5("Model Workflow"),
                           # verbatimTextOutput('compare_model_info')
                 ),
                 nav_panel("Results", 
                           h5("10-fold Cross-validation Results"),
                           uiOutput('cv_results_text_description_compare'),
                           DTOutput("compare_cv_results")
                           # uiOutput('results_pred_explanation'),
                           # DTOutput("results")
                           ),
                 nav_panel("Plot",
                           uiOutput('compare_model_metric'),
                           plotOutput("compare_model_plot"))
               )
             )
           )
  ),
  tabPanel("Single Model Evaluation",
           sidebarLayout(
           sidebarPanel(
             width = 3,
             selectInput("dataset_choice", "Select Dataset", choices = c("Palmer Penguins", "Boston Housing", "Upload Dataset"),
                         selected = 'Palmer Penguins'),
             
             # Conditional file input, visible only if 'Upload Dataset' is selected
             conditionalPanel(
               condition = "input.dataset_choice == 'Upload Dataset'",
               fileInput("dataset", "Upload .csv File", accept = c(".csv"))
             ),
             uiOutput('target'),
             uiOutput("select_columns"),
             uiOutput("select_categorical"),
             uiOutput('ml_type'),
             uiOutput('model'),
             checkboxGroupInput("preproc", 'Data Preprocessing',
                                c("Yeo-Johnson Transformation" = 'yeo_johnson',
                                  "Center" = 'center',
                                  "Scale" = 'scale',
                                  "Impute NA's" = 'impute'
                                )),
             conditionalPanel(
               condition = "$.inArray('impute', input.preproc) > -1",
               radioButtons("impute_type", 'Imputation Method',
                            c("Median" = 'med',
                              "Mean" = 'mean',
                              "KNN" = 'knn',
                              "Linear" = 'linear'
                            ))
             ),
             conditionalPanel(
               condition = "$.inArray('impute', input.preproc) > -1",
               radioButtons("cat_impute", "Impute Categorical NA's?",
                            c("Yes", "No"), selected = 'No')
             ),
             radioButtons("tuning", "Model Tuning", choices = c("Automatic", "Default Parameters"), selected = 'Default Parameters'),
             conditionalPanel(
               condition = "input.tuning == 'Automatic'",
               uiOutput('model_metric')
             ),
             checkboxInput('parallel', label = "Enable Parallelization?", value = FALSE),
             conditionalPanel(
               condition = "input.parallel == true",
               numericInput('num_cores', label = 'Number of Cores', value = 2, min = 2, max = parallel::detectCores())
             ),
             actionButton("train", "Train Model")
             
           ),
           mainPanel(
             width = 9,
               htmlOutput('model_selected_text'),
               
               navset_card_underline(
                 nav_panel('Data', DTOutput("data_table"),
                           # tags$hr(),
                           p(''),
                           uiOutput('data_rows'),
                           br(),
                           uiOutput('data_cols')),
                 nav_panel('Model Info',
                           p('Note the following:'),
                           HTML('<ul><li>The dataset is split into train and test sets (80/20 split) for model training and evaluation.</li><li>Cross-validation is performed using the 80% training set to assess model performance.</li></ul>'),
                           uiOutput('model_info_variables'),
                           h5("Model Workflow"),
                           verbatimTextOutput('model_info')),
                 nav_panel("Results", 
                           h5("5-fold Cross-validation Results (using the 80% training set split)"),
                           uiOutput('cv_results_text_description'),
                           verbatimTextOutput("cv_results"),
                           uiOutput('results_pred_explanation'),
                           DTOutput("results")),
                 nav_panel("Plot", 
                           h5("Plot of Prediction Results (using the 20% testing set split)"),
                           plotOutput("model_plot"))
               )
           )
           )
  ),
  tabPanel('Prediction', 'This page under construction.')
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive expression for reading the dataset
  data <- reactive({
    if (input$dataset_choice == 'Palmer Penguins') {
      penguins # Use palmerpenguins dataset as default
    } else if(input$dataset_choice == 'Boston Housing') {
      Boston
    }else if(input$dataset_choice == 'Upload Dataset') {
      if(!is.null(input$dataset)) {
        read.csv(input$dataset$datapath)
      } else {
        penguins
      }
    }
  })
  
  # Update UI for column selection based on uploaded dataset
  # observe({
  #   req(data())
  #   updateSelectInput(session, inputId = "target", choices = names(data()))
  # })
  

# Compare Models ----------------------------------------------------------

  output$compare_target <- renderUI({
    req(data())
    selectInput("compare_target", "Select Target Variable", choices = names(data()))
  })
  
  output$compare_select_columns <- renderUI({
    checkboxGroupInput("compare_columns", "Select All Predictors", choices = names(data())[which(names(data()) != input$compare_target)], selected = names(data()))
  })
  
  output$compare_select_categorical <- renderUI({
    checkboxGroupInput("compare_categorical", "Select Categorical Predictors", choices = names(data())[which(names(data()) %in% input$compare_columns)], selected = NULL)
  })
  
  output$compare_data_rows <- reactive(
    paste('Total number of rows:', nrow(data()))
  )
  
  output$compare_data_cols <- reactive(
    paste('Total number of columns:', ncol(data()))
  )
  
  output$compare_data_table <- renderDT({
    data()
  }, fillContainer = TRUE)
  
  output$compare_ml_type <- renderUI({
    req(data(), input$compare_target)
    df <- data()
    target_type <- df |> pull(input$compare_target) |> class()
    if(target_type == 'factor' | target_type == 'logical') {
      selected_type = 'classification'
    } else {
      selected_type = 'regression'
    }
    radioButtons('compare_ml_type', 'Model Type', choices = c("Regression" = 'regression',
                                                              "Classification" = 'classification'),
                 selected = selected_type, inline = TRUE)
  })
  
  # output$compare_model_info <- renderPrint({
  #   req(compare_model_results())
  #   model_results()$workflow
  # })
  
  cv_results_text_description_event_compare <- eventReactive(input$compare_train, {
    req(compare_model_results())
    if(input$compare_ml_type == 'regression') {
      metrics_use <- c('rmse', 'mae', 'rsq')
    } else {
      metrics_use <- c('accuracy', 'precision', 'recall', 'sens', 'spec', 'roc_auc', 'f_meas') 
    }
    models <- c()
    num_models <- length(input$compare_model)
    for(i in 1:num_models) {
      if(i < num_models) {
        models <- paste0(models, '<code>', input$compare_model[i], '</code>', ', ')
      } else {
        models <- paste0(models, '<code>', input$compare_model[i], '</code>', '.')
      }
    }
    paste(paste0("<div>The table below displays the 10-fold CV performance (using <b>default parameters</b>) of the following models: ", models, "</div><ul><li>10-fold CV took ", round(compare_model_results()$fit_time, 2), " seconds from start to finish using ", compare_model_results()$num_cores, ".</li></ul>"),
          paste0('<div>The following metrics are reported:', paste0('<code>', metrics_use, collapse = '</code>, '), '</code>.</div>'))
  })
  
  output$cv_results_text_description_compare <- renderUI({
    HTML(cv_results_text_description_event_compare())
  })
  
  compare_model_info_variables_event <- eventReactive(input$compare_train, {
    req(compare_model_results())
    predictors <- c()
    num_preds <- length(input$compare_columns)
    for(i in 1:num_preds) {
      if(i < num_preds) {
        predictors <- paste0(predictors, '<code>', input$compare_columns[i], '</code>', ', ')
      } else {
        predictors <- paste0(predictors, '<code>', input$compare_columns[i], '</code>', '.')
      }
    }
    paste0(
      '<div>Selected target variable: <code style="display:inline;">', input$compare_target, '</code>.</div>',
      '<div>Selected predictor variables:', predictors, '</div>')
  })
  
  output$compare_model_info_variables <- renderUI({
    HTML(compare_model_info_variables_event())
  })
  
  output$compare_model <- renderUI({
    req(data(), input$compare_target, input$compare_ml_type)
    
    if(input$compare_ml_type == 'regression') {
      model_choices <- c("Linear Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
    } else {
      num_classes <- data() |> pull(input$compare_target) |> unique() |> length()
      if(num_classes == 2) {
        model_choices <- c("Logistic Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
      } else {
        model_choices <- c("Multinomial Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
      }
      
    }
    checkboxGroupInput("compare_model", "Select Models", choices = model_choices)
  })
  
  # Reactive expression for model specification and training
  compare_model_results <- eventReactive(input$compare_train, {
    req(input$compare_target)
    showNotification("Fitting models...", type = "message")
    tictoc::tic()
    df <- data()
    
    df <- df[, c(input$compare_columns, input$compare_target), drop = FALSE]
    
    # If not imputing missingness, drop all rows with missing values
    if(!("impute" %in% input$preproc)) {
      df <- df |> drop_na()
    }
    
    # Convert selected categorical columns to factors
    df[input$compare_categorical] <- lapply(df[input$compare_categorical], as.factor)
    
    # Convert all non-categorical columns to numerical
    numeric_cols <- setdiff(input$compare_columns, input$compare_categorical)
    df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)
    
    train_data <- df
    
    # Define preprocessing recipe
    rec <- recipe(as.formula(paste(input$compare_target, "~ .")), data = train_data) |> 
      step_dummy(all_factor_predictors())
    
    # Add additional preprocessing steps
    if ("yeo_johnson" %in% input$preproc) {
      rec <- rec |> 
        step_YeoJohnson(all_numeric_predictors())
    }
    if ("scale" %in% input$preproc) {
      rec <- rec |> 
        step_scale(all_numeric_predictors())
    }
    if ("center" %in% input$preproc) {
      rec <- rec |> 
        step_center(all_numeric_predictors())
    }
    if ("impute" %in% input$preproc) {
      if(input$impute_type == 'median') {
        rec <- rec |> 
          step_impute_median(all_numeric_predictors())
      } else if (input$impute_type == 'mean') {
        rec <- rec |> 
          step_impute_mean(all_numeric_predictors())
      } else if (input$impute_type == 'knn') {
        rec <- rec |> 
          step_impute_knn(all_numeric_predictors())
      } else if (input$impute_type == 'linear') {
        rec <- rec |> 
          step_impute_linear(all_numeric_predictors())
      }
      if (input$cat_impute == 'Yes') {
        rec <- rec |> 
          step_impute_bag(all_factor_predictors())
      }
    }
    
    ml_type <- input$compare_ml_type
    
    model_list <- list()
    if ("Logistic Regression" %in% input$compare_model) {
      lr_model_spec <- logistic_reg() |>
        set_engine("glm") |>
        set_mode(ml_type)
      model_list$lr <- lr_model_spec
    }
    if ("Multinomial Regression" %in% input$compare_model) {
      mr_model_spec <- multinom_reg(penalty = 0) |>
        set_engine("glmnet") |>
        set_mode(ml_type)
      model_list$mr <- mr_model_spec
    }
    if ("Linear Regression" %in% input$compare_model) {
      lr_model_spec <- linear_reg() |>
        set_engine("glm") |>
        set_mode(ml_type)
      model_list$lr <- lr_model_spec
    }
    if ("KNN" %in% input$compare_model) {
      knn_model_spec <- nearest_neighbor() |>
        set_engine("kknn") |>
        set_mode(ml_type)
      model_list$knn <- knn_model_spec
    }
    if ("Random Forest" %in% input$compare_model) {
      rf_model_spec <- rand_forest() |>
        set_engine("randomForest") |>
        set_mode(ml_type)
      model_list$rf <- rf_model_spec
    }
    if ("XGBoost" %in% input$compare_model) {
      xgb_model_spec <- boost_tree() |>
        set_engine('xgboost') |>
        set_mode(ml_type)
      model_list$xgb <- xgb_model_spec
    }
    if ("Neural Network" %in% input$compare_model) {
      nnet_model_spec <- mlp() |>
        set_engine("nnet") |>
        set_mode(ml_type)
      model_list$nnet <- nnet_model_spec
    }
    if ("Support Vector Machine" %in% input$compare_model) {
      svm_rbf_spec <- svm_rbf() |>
        set_engine('kernlab') |>
        set_mode(ml_type)
      model_list$svm_rbf <- svm_rbf_spec
    }
    
    # Workflow
    workflow <- workflow_set(preproc = list(all_preproc = rec),
                             models = model_list)
    
    # 10-fold CV
    folds <- vfold_cv(train_data, v = 10)
    
      if(ml_type == 'regression') {
        metric_set_use <- metric_set(rmse, mae, rsq)
      } else {
        metric_set_use <- metric_set(accuracy, precision, recall, sens, spec, roc_auc, f_meas) 
      }
      
      # Set up parallel processing to expedite evaluation (this doesn't work on a free shinyapps.io hosted app)
      if(input$parallel == TRUE) {
        plan(multisession, workers = input$num_cores)
      }
      
      cv_fit <- workflow_map(workflow,
                             fn = 'fit_resamples',
                             resamples = folds,
                             metrics = metric_set_use)
      
      cv_results <- collect_metrics(cv_fit) |> 
        select(model, .metric, mean) |>
        arrange(mean) |>
        mutate(model = case_when(model == 'mlp' ~ "Neural Network",
                                 model == 'boost_tree' ~ "XGBoost",
                                 model == 'multinom_reg' ~ "Multinomial Regression",
                                 model == 'linear_reg' ~ "Linear Regression",
                                 model == 'rand_forest' ~ "Random Forest",
                                 model == 'nearest_neighbor'~ "KNN",
                                 model == 'logistic_reg' ~ "Logistic Regression",
                                 model == 'svm_rbf' ~ "Support Vector Machine")) |> 
        rename(Metric = .metric, Value = mean)
      
      if(input$parallel == TRUE) {
        plan(sequential)
      }
    
    showNotification("Model Fitting complete!", type = "message")
    
    toc <- tictoc::toc()
    num_cores <- ifelse(input$parallel, paste(input$num_cores, 'cores'), '1 core')
    list(workflow = workflow, cv_results = cv_results,
         fit_time = toc$toc - toc$tic, num_cores = num_cores, cv_fit = cv_fit)
    
  })
  
  output$compare_cv_results <- renderDT({
    req(compare_model_results())
    compare_model_results()$cv_results
  }, fillContainer = TRUE)
  
  output$compare_model_metric <- renderUI({
    if(input$compare_ml_type == 'regression') {
      metric_choices <- c("RMSE" = 'rmse', "MAE" = 'mae', "R Squared" = 'rsq')
    } else {
      metric_choices <- c("Accuracy" = 'accuracy', "Precision" = 'precision', "Recall" = 'recall',
                          "Sensitivity" = 'sens', "Specificity" = 'spec',
                          "Recall" = 'recall', "F1 Score" = 'f_meas',
                          "ROC AUC" = 'roc_auc')
    }
    radioButtons("compare_model_metric", "Plotting Metric", choices = metric_choices, selected = metric_choices[1], inline = TRUE)
  })
  
  output$compare_model_plot <- renderPlot({
    req(compare_model_results())
    
    # Visualize performance of best models
    theme_set(theme_light())
    cv_fit <- compare_model_results()$cv_fit
    cv_fit$info <- cv_fit$info
    change_fit_names <- function(x) {
      x[, 3] <- case_when(x[, 3] == 'mlp' ~ "Neural Network",
                        x[, 3] == 'boost_tree' ~ "XGBoost",
                        x[, 3] == 'multinom_reg' ~ "Multinomial Regression",
                        x[, 3] == 'linear_reg' ~ "Linear Regression",
                        x[, 3] == 'rand_forest' ~ "Random Forest",
                        x[, 3] == 'nearest_neighbor'~ "KNN",
                        x[, 3] == 'logistic_reg' ~ "Logistic Regression",
                        x[, 3] == 'svm_rbf' ~ "Support Vector Machine")
      x
    }
    cv_fit$info <- cv_fit$info |> map(change_fit_names)
    
    autoplot(cv_fit,
             rank_metric = input$compare_model_metric,
             metric = input$compare_model_metric)
    
  })
  
# Single Model Evaluation -------------------------------------------------

  output$target <- renderUI({
    req(data())
    selectInput("target", "Select Target Variable", choices = names(data()))
  })
  
  output$select_columns <- renderUI({
    checkboxGroupInput("columns", "Select All Predictors", choices = names(data())[which(names(data()) != input$target)], selected = names(data()))
  })
  
  output$select_categorical <- renderUI({
    checkboxGroupInput("categorical", "Select Categorical Predictors", choices = names(data())[which(names(data()) %in% input$columns)], selected = NULL)
  })
  
  output$ml_type <- renderUI({
    req(data(), input$target)
    df <- data()
    target_type <- df |> pull(input$target) |> class()
    if(target_type == 'factor' | target_type == 'logical') {
      selected_type = 'classification'
    } else {
      selected_type = 'regression'
    }
    radioButtons('ml_type', 'Model Type', choices = c("Regression" = 'regression',
                                                      "Classification" = 'classification'),
                 selected = selected_type, inline = TRUE)
  })
  
  output$model <- renderUI({
    req(data(), input$target, input$ml_type)
    
    if(input$ml_type == 'regression') {
      model_choices <- c("Linear Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
    } else {
      num_classes <- data() |> pull(input$target) |> unique() |> length()
      if(num_classes == 2) {
        model_choices <- c("Logistic Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
      } else {
        model_choices <- c("Multinomial Regression", "KNN", "Random Forest", "XGBoost", "Neural Network", "Support Vector Machine")
      }
      
    }
    selectInput("model", "Select Model", choices = model_choices)
  })
  
  output$model_metric <- renderUI({
    if(input$ml_type == 'regression') {
      metric_choices <- c("RMSE" = 'rmse', "MAE" = 'mae', "R Squared" = 'rsq')
    } else {
      metric_choices <- c("Accuracy" = 'accuracy', "Precision" = 'precision', "Recall" = 'recall',
                          "Sensitivity" = 'sens', "Specificity" = 'spec',
                          "Recall" = 'recall', "F1 Score" = 'f_meas',
                          "ROC AUC" = 'roc_auc')
    }
    selectInput("model_metric", "Model Metric", choices = metric_choices)
  })
  
  # Reactive expression for model specification and training
  model_results <- eventReactive(input$train, {
    req(input$target)
    showNotification("Training started...", type = "message")
    tictoc::tic()
    df <- data()
    
    df <- df[, c(input$columns, input$target), drop = FALSE]
    
    # If not imputing missingness, drop all rows with missing values
    if(!("impute" %in% input$preproc)) {
      df <- df |> drop_na()
    }
    
    # Convert selected categorical columns to factors
    df[input$categorical] <- lapply(df[input$categorical], as.factor)
    
    # Convert all non-categorical columns to numerical
    numeric_cols <- setdiff(input$columns, input$categorical)
    df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)
    
    # Split data
    split <- initial_split(df, prop = 0.8, strata = input$target)
    train_data <- training(split)
    test_data <- testing(split)
    
    # Define preprocessing recipe
    rec <- recipe(as.formula(paste(input$target, "~ .")), data = train_data) |> 
      step_dummy(all_factor_predictors())
    
    # Add additional preprocessing steps
    if ("yeo_johnson" %in% input$preproc) {
      rec <- rec |> 
        step_YeoJohnson(all_numeric_predictors())
    }
    if ("scale" %in% input$preproc) {
      rec <- rec |> 
        step_scale(all_numeric_predictors())
    }
    if ("center" %in% input$preproc) {
      rec <- rec |> 
        step_center(all_numeric_predictors())
    }
    if ("impute" %in% input$preproc) {
      if(input$impute_type == 'median') {
        rec <- rec |> 
          step_impute_median(all_numeric_predictors())
      } else if (input$impute_type == 'mean') {
        rec <- rec |> 
          step_impute_mean(all_numeric_predictors())
      } else if (input$impute_type == 'knn') {
        rec <- rec |> 
          step_impute_knn(all_numeric_predictors())
      } else if (input$impute_type == 'linear') {
        rec <- rec |> 
          step_impute_linear(all_numeric_predictors())
      }
      if (input$cat_impute == 'Yes') {
        rec <- rec |> 
          step_impute_bag(all_factor_predictors())
      }
    }
    
    ml_type <- input$ml_type
    
    if (input$model == "Logistic Regression") {
      if(input$tuning == 'Automatic') {
        model_spec <- logistic_reg(penalty = tune(),
                                   mixture = tune()) |>
          set_engine("glmnet") |>
          set_mode(ml_type)
      } else {
        model_spec <- logistic_reg() |>
          set_engine("glm") |>
          set_mode(ml_type)
      }
    } else if (input$model == "Multinomial Regression") {
      if(input$tuning == 'Automatic') {
        model_spec <- multinom_reg(penalty = tune(),
                                   mixture = tune()) |>
          set_engine("glmnet") |>
          set_mode(ml_type)
      } else {
        model_spec <- multinom_reg(penalty = 0) |>
          set_engine("glmnet") |>
          set_mode(ml_type)
      }
    } else if (input$model == "Linear Regression") {
      if(input$tuning == 'Automatic') {
        model_spec <- linear_reg(penalty = tune(),
                                 mixture = tune()) |>
          set_engine("glmnet") |>
          set_mode(ml_type)
      } else {
        model_spec <- linear_reg() |>
          set_engine("glm") |>
          set_mode(ml_type)
      }
    } else if (input$model == "KNN") {
      if(input$tuning == 'Automatic') {
        model_spec <- nearest_neighbor(neighbors = tune(),
                                       weight_func = tune()) |> 
          set_engine("kknn") |> 
          set_mode(ml_type)
      } else {
        model_spec <- nearest_neighbor() |> 
          set_engine("kknn") |> 
          set_mode(ml_type)
      }
    } else if (input$model == "Random Forest") {
      if(input$tuning == 'Automatic') {
        model_spec <- rand_forest(mtry = tune(),
                                  trees = tune(),
                                  min_n = tune()) |>
          set_engine("randomForest") |>
          set_mode(ml_type)
      } else {
        model_spec <- rand_forest() |>
          set_engine("randomForest") |>
          set_mode(ml_type)
      }
    } else if (input$model == 'XGBoost') {
      if(input$tuning == 'Automatic') {
        model_spec <- boost_tree(tree_depth = tune(),
                                 learn_rate = tune(),
                                 min_n = tune(),
                                 sample_size = tune(),
                                 trees = tune()) |>
          set_engine("xgboost") |>
          set_mode(ml_type)
      } else {
        model_spec <- boost_tree() |>
          set_engine('xgboost') |>
          set_mode(ml_type)
      }
    } else if (input$model == 'Neural Network') {
      if(input$tuning == 'Automatic') {
        model_spec <- mlp(hidden_units = tune(),
                          penalty = tune(),
                          epochs = tune()) |>
          set_engine("nnet") |>
          set_mode(ml_type)
      } else {
        model_spec <- mlp() |>
          set_engine("nnet") |>
          set_mode(ml_type)
      }
    } else if (input$model == "Support Vector Machine") {
      if(input$tuning == 'Automatic') {
        model_spec <- svm_rbf(cost = tune(),
                                rbf_sigma = tune()) |>
          set_engine('kernlab') |>
          set_mode(ml_type)
      } else {
        model_spec <- svm_rbf() |>
          set_engine('kernlab') |>
          set_mode(ml_type)
      }
    }
    
    # Workflow
    workflow <- workflow()  |>
      add_model(model_spec) |>
      add_recipe(rec)
    
    # -fold CV
    folds <- vfold_cv(train_data, v = 5)
    
    
    if(input$tuning == 'Default Parameters') {
      
      if(ml_type == 'regression') {
        metric_set_use <- metric_set(rmse, mae, rsq)
      } else {
        metric_set_use <- metric_set(accuracy, precision, recall, sens, spec, roc_auc, f_meas) 
      }
      
      # Set up parallel processing to expedite evaluation (this doesn't work on a free shinyapps.io hosted app)
      if(input$parallel == TRUE) {
        plan(multisession, workers = input$num_cores)
      }
      
      cv_fit <- fit_resamples(workflow, folds, metrics = metric_set_use)
      
      cv_results <- collect_metrics(cv_fit, type = 'wide') #|> arrange(.metric == input$model_metric, mean)
      
      # Fit model
      fit <- workflow |> fit(train_data)
      
      if(input$parallel == TRUE) {
        plan(sequential)
      }
      
    } else if(input$tuning == 'Automatic') {
      
      # Set up parallel processing to expedite tuning (this doesn't work on a free shinyapps.io hosted app)
      if(input$parallel == TRUE) {
        plan(multisession, workers = input$num_cores)
      }
      
      # Execute workflow, obtaining 5-fold CV metric values across all tuning parameter combinations
      if(input$model == 'Support Vector Machine') {
        model_params <- workflow |> 
          extract_parameter_set_dials()
      } else {
        model_params <- workflow |> 
          extract_parameter_set_dials() |> 
          finalize(train_data |> select(!input$target))
      }
      if(ml_type == 'regression') {
        metric_set_use <- metric_set(rmse, mae, rsq)
      } else {
        metric_set_use <- metric_set(accuracy, precision, recall, sens, spec, roc_auc, f_meas) 
      }
      model_tune <- workflow |> 
        tune_grid(folds,
                  grid = model_params |> grid_regular(levels = 3),
                  metrics = metric_set_use)
      
      cv_results <- model_tune |>
        show_best(metric = input$model_metric) |>
        select(!c(.estimator, n, .config))
      
      best_tune <- model_tune |> select_best(metric = input$model_metric)
      
      # Get final model with with the best model
      fit <- workflow |> 
        finalize_workflow(best_tune) |> 
        fit(train_data)

      if(input$parallel == TRUE) {
        plan(sequential)
      }
    }
    
    # Predict and evaluate
    results <- augment(fit, test_data) |> relocate(input$target, .before = 1)
    # bind_cols(test_data) |>
    # metrics(truth = !!sym(input$target), estimate = .pred_class)
    
    showNotification("Training complete!", type = "message")
    
    toc <- tictoc::toc()
    num_cores <- ifelse(input$parallel, paste(input$num_cores, 'cores'), '1 core')
    list(fit = fit, results = results, workflow = workflow, cv_results = cv_results,
         test_data = test_data, fit_time = toc$toc - toc$tic, num_cores = num_cores)
    
  })
  
  
  # Display dataset
  output$data_table <- renderDT({
    data()
  }, fillContainer = TRUE)
  
  output$data_rows <- reactive(
    paste('Total number of rows:', nrow(data()))
  )
  
  output$data_cols <- reactive(
    paste('Total number of columns:', ncol(data()))
  )
  
  # Display model selected
  model_selected_text_event <- eventReactive(input$train, {
    paste("<div style='font-size: 22px;'>", 
          "Model selected: <b>", input$model, "</b>",
          "</div>")
  })
  output$model_selected_text <- renderUI({
    HTML(model_selected_text_event())
  })
  
  output$model_info <- renderPrint({
    req(model_results())
    model_results()$workflow
  })
  
  model_info_variables_event <- eventReactive(input$train, {
    req(model_results())
    predictors <- c()
    num_preds <- length(input$columns)
    for(i in 1:num_preds) {
      if(i < num_preds) {
        predictors <- paste0(predictors, '<code>', input$columns[i], '</code>', ', ')
      } else {
        predictors <- paste0(predictors, '<code>', input$columns[i], '</code>', '.')
      }
    }
    paste0(
      '<div>Selected target variable: <code style="display:inline;">', input$target, '</code>.</div>',
      '<div>Selected predictor variables:', predictors, '</div>')
  })
  
  output$model_info_variables <- renderUI({
    HTML(model_info_variables_event())
  })
  
  output$cv_results <- renderPrint({
    req(model_results())
    model_results()$cv_results
    
  })
  
  cv_results_text_description_event <- eventReactive(input$train, {
    req(model_results())
    if(input$ml_type == 'regression') {
      metrics_use <- c('rmse', 'mae', 'rsq')
    } else {
      metrics_use <- c('accuracy', 'precision', 'recall', 'sens', 'spec', 'roc_auc', 'f_meas') 
    }
    if(input$tuning == 'Automatic') {
      best_score <- model_results()$cv_results[1, ]$mean
      paste(paste0("<div>The table below displays the performance of the 5 best <b>",
                   input$model,
                   "</b> models using various combinations of tuning parameters.</div>"),
            paste0("<ul><li>Tuning was performed to optimize <code>", input$model_metric, "</code>.</li><li>5-fold CV took ", round(model_results()$fit_time, 2), " seconds from start to finish using ", model_results()$num_cores, ".</li><li>The best model achieves the following 5-fold CV <code>", input$model_metric, "</code>: ", round(best_score, 4), ".</li></ul>"))
    } else {
      paste(paste0("<div>The table below displays the 5-fold CV performance of the <b>", input$model, "</b> model using <b>default parameters</b>.</div><ul><li>5-fold CV took ", round(model_results()$fit_time, 2), " seconds from start to finish using ", model_results()$num_cores, ".</li></ul>"),
            paste0('<div>The following metrics are reported:', paste0('<code>', metrics_use, collapse = '</code>, '), '</code>.</div>'))
    }
  })
  output$cv_results_text_description <- renderUI({
    HTML(cv_results_text_description_event())
  })
  
  results_pred_explanation_event <- eventReactive(input$train, {
    req(model_results())
    paste('The table below shows the (20%) testing set along with model predictions.
         <br>
         <ul><li>Observed values of <code>', input$target, '</code> are given in the first column.
         </li><li>Predictions are indicated with <code> .pred</code>.</li></ul>')
  })
  
  output$results_pred_explanation <- renderUI({
    HTML(results_pred_explanation_event())
  })
  
  output$results <- renderDT({
    req(model_results())
    model_results()$results
  })
  
  
  # Display plot
  output$model_plot <- renderPlot({
    req(model_results())
    
    results <- model_results()$results
    test_data <- model_results()$test_data
    
    if (input$ml_type == 'classification') {
      cm <- caret::confusionMatrix(results$.pred_class, test_data |> pull(input$target))
      
      # Convert confusion matrix to data frame for ggplot2
      cm_df <- as.data.frame(cm$table)
      
      cm_df <- cm_df %>%
        group_by(Reference) %>%
        mutate(Proportion = Freq / sum(Freq))  # Normalize by row for percent correct
      
      # Plot confusion matrix as heatmap with counts and percent correct
      ggplot(cm_df, aes(x = Prediction, y = Reference)) +
        geom_tile(aes(fill = Proportion), color = "white") +
        geom_text(aes(label = paste0(Freq, "\n(", percent(Proportion), ")")), vjust = 1, color = "black") +
        scale_fill_gradient(low = "white", high = "steelblue") +
        labs(title = paste0('Confusion Matrix for "', input$target, '"'), x = "Predicted Class", y = "Actual Class") +
        theme_minimal() +
        theme(
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          plot.title = element_text(size = 14, face = "bold")
        )
    } else {
      true_values <- model_results()$test_data %>% pull(input$target)
      predictions <- model_results()$results$.pred
      
      # Create a data frame for plotting
      plot_data <- data.frame(
        True = true_values,
        Predicted = predictions
      )
      
      # Plot predicted vs. true values
      ggplot(plot_data, aes(x = True, y = Predicted)) +
        geom_point(alpha = 0.6) +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
        labs(title = "Predicted vs. True Values",
             x = "True Values",
             y = "Predicted Values") +
        theme_minimal() +
        theme(
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          plot.title = element_text(size = 14, face = "bold")
        )
    }
  })
}

# Run the application 
shinyApp(ui = ui, server = server)


# If writing manifest file to publish app via Posit connect:
# rsconnect::writeManifest(appFiles = 'ml_shiny.R', appMode = 'shiny')


# Things to do:

# - Plot showing model info (like variable importance for RF/XGBoost)
# - Add functionality to allow users to obtain prediction(s) on new data using the trained model
# in cv results, "model" = "Model" in tibble name, also rename the metrics to their full names (like I did with the)