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

# Define UI for the application
ui <- navbarPage(
  title = "Shiny App for Machine Learning",
  theme = bs_theme(version = 5, preset = 'cosmo'),
  
  tabPanel("Model Evaluation",
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
      model_choices <- c("Linear Regression", "KNN", "Random Forest", "XGBoost")
    } else {
      num_classes <- data() |> pull(input$target) |> unique() |> length()
      if(num_classes == 2) {
        model_choices <- c("Logistic Regression", "KNN", "Random Forest", "XGBoost")
      } else {
        model_choices <- c("Multinomial Regression", "KNN", "Random Forest", "XGBoost")
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
    }
    
    # Workflow
    workflow <- workflow()  |>
      add_model(model_spec) |>
      add_recipe(rec)
    
    # 5-fold CV
    folds <- vfold_cv(train_data, v = 5)
    
    
    if(input$tuning == 'Default Parameters') {
      
      if(ml_type == 'regression') {
        metric_set_use <- metric_set(rmse, mae, rsq)
      } else {
        metric_set_use <- metric_set(accuracy, precision, recall, sens, spec, roc_auc, f_meas) 
      }
      
      cv_fit <- fit_resamples(workflow, folds, metrics = metric_set_use)
      
      cv_results <- collect_metrics(cv_fit, type = 'wide') #|> arrange(.metric == input$model_metric, mean)
      
      # Fit model
      fit <- workflow |> fit(train_data)
      
    } else if(input$tuning == 'Automatic') {
      
      # Set up parallel processing to expedite tuning (this doesn't work on a free shinyapps.io hosted app)
      # library(doParallel)
      # cl <- makePSOCKcluster(4)
      # registerDoParallel(cl)
      
      # Execute workflow, obtaining 5-fold CV metric values across all tuning parameter combinations
      model_params <- workflow |> 
        extract_parameter_set_dials() |> 
        finalize(train_data |> select(!input$target))
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
    }
    
    # Predict and evaluate
    results <- augment(fit, test_data) |> relocate(input$target, .before = 1)
    # bind_cols(test_data) |>
    # metrics(truth = !!sym(input$target), estimate = .pred_class)
    
    showNotification("Training complete!", type = "message")
    
    list(fit = fit, results = results, workflow = workflow, cv_results = cv_results,
         test_data = test_data)
    
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
            paste0("<ul><li>Tuning was performed to optimize <code>", input$model_metric, "</code>.</li><li>The best model achieves the following 5-fold CV <code>", input$model_metric, "</code>: ", round(best_score, 4), ".</li></ul>"))
    } else {
      paste(paste0("<div>The table below displays the 5-fold CV performance of the <b>", input$model, "</b> model using <b>default parameters</b>.</div>"),
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
  
  
  # Display plot (example: ROC curve for logistic regression)
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


# If writing manifest file to publish app via Posit connect 
# rsconnect::writeManifest(appFiles = 'ml_shiny.R', appMode = 'shiny')



# Things to do:

# - Plot showing model info (like variable importance for RF/XGBoost)
# - Add functionality to allow users to obtain prediction(s) on new data using the trained model
# - on the results page, display the time it took to do CV
# - host the app on github with free posit connect. That way, people can easily find the code to
# download and run locally if they want
