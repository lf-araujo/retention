# Load the glmnet package
library(glmnet)

# register parallel backend
library(doParallel)
# make cluster with all available clusters
cl <- makeCluster(detectCores(), outfile = "")
registerDoParallel(cl)

# Load the data
retention <- read.csv("~/Downloads/dataIndiv_ID_2023-03-08.csv")

# Set outcome variable and covariates
y <- retention$visit_status_aggr
# visit_status_aggr
X <- model.matrix(~ income_household + site, data = retention)

# Fit LASSO model with cross-validation
fit <- cv.glmnet(X, y, alpha = 1, family = "multinomial", parallel = T)

# Plot cross-validation results
plot(fit)

# Identify the lambda value with the lowest cross-validation error
best_lambda <- fit$lambda.1se

# Fit final LASSO model with the selected lambda value
lasso_model <- glmnet(X, y, alpha = 1, lambda = best_lambda, family = "multinomial", parallel = T)

# Extract coefficients from the final LASSO model
lasso_coef <- coef(lasso_model)

# Identify the most influential covariates (non-zero coefficients)
influential_covariates <- rownames(lasso_coef[[1]])[which(lasso_coef[[1]] != 0)]

# Print the results
cat("Most influential covariates:\n")

for (i in 1:length(influential_covariates)) {
  cat("Covariate:", influential_covariates[i], "\n")
  cat("Coefficients:")
  for (j in 1:ncol(lasso_coef[[1]])) {
    cat(paste(round(lasso_coef[[1]][influential_covariates[i], j], 2), " ", sep = ""))
  }
  cat("\n\n")
}


# duration_home_site30 min to 60 min
# duration_home_site60 min to 90 min
# duration_home_siteMore than 90 min
# duration_home_siteUnknown
# eduDon't Know
# eduup to Bachelor's degree
# eduup to High school graduate
# emplDisabled
# emplMaternity leave
# emplOther
# emplRefuse to answer
# emplRetired
# emplSick leave
# emplStay at home parent
# emplStudent
# emplTemporarily Laid off
# emplUnemployed, looking
# emplUnemploymed, not looking
# genderFemale
# genderIntersex−Male
# hhinc<50k
# hhinc50k−100k
# hhincUnknown
# mri_report_action
# race_ethnAsian
# race_ethnBlack
# race_ethnHispanic
# race_ethnOther
# race_ethnUnknown
# siteCHLA
# siteCUB
# siteFIU
# siteLIBR
# siteMUSC
# siteOHSU
# siteROC
# siteSRI
# siteUCLA
# siteUCSD
# siteUFL
# siteUMB
# siteUMICH
# siteUMN
# siteUTAH
# siteUVM
# siteUWM
# siteVCU
# siteWUSTL
# siteYALE
# spanishyes
# substudy_fitbit___1
# substudy_prodromal_psych___1
# su



# Site
# income_household  or income_household_aggr
# education_household_aggr (but check collinearity with income)
# race_ethnicity (but probably replace this with ethnicity from Parent Demographics when we have it)
