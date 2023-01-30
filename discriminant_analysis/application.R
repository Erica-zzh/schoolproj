#' ---
#' title: "hw4_starter.R (Problem 4 and Problem 5)"
#' author: "Erica Zhou"
#' date: "Nov 30, 2022"
#' ---

#' Problem 4

rm(list = ls())  # remove the existing environment

## You should set the working directory to the folder of hw3_starter by
## uncommenting the following and replacing YourDirectory by what you have
## in your local computer / labtop

setwd("~/STA314/sta314-hw4")

## Load utils.R and discriminant_analysis.R

source("utils.R")
source("discriminant_analysis.R")


## Load the training and test data
train <- Load_data("digits_train.txt")
test <- Load_data("digits_test.txt")

x_train <- train$x
y_train <- train$y

x_test <- test$x
y_test <- test$y




#####################################################################
#                           Part a.                                 #
# TODO:  estimate the priors, conditional means and conditional     #
#        covariance matrices under LDA,                             #
#        predict the labels of test data by using the fitted LDA    #
#        compute its misclassification error rate                   #
#####################################################################

priors <- Comp_priors(y_train)
means <- Comp_cond_means(x_train,y_train)
covs <- Comp_cond_covs(x_train,y_train, TRUE)
post <- Predict_posterior(x_test,priors,means,covs,TRUE)
post_label <- Predict_labels(post)

# the error rate is 
mean(y_test != post_label)

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################



#####################################################################
#                           Part b.                                 #
# TODO:  estimate the priors, conditional means and conditional     #
#        covariance matrices under QDA,                             #
#        predict the labels of test data by using the fitted LDA    #
#        compute its misclassification error rate                   #
#####################################################################

priors <- Comp_priors(y_train)
means <- Comp_cond_means(x_train,y_train)
covs <- Comp_cond_covs(x_train,y_train, FALSE)
post <- Predict_posterior(x_test,priors,means,covs,FALSE)
post_label <- Predict_labels(post)

# the error rate is 
mean(y_test != post_label)

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################



#####################################################################
#                           Part c.                                 #
# TODO:  fit LDA and QDA by using the R package                     #
#        report their test errors                                   #
#####################################################################

library(MASS)

lda.fit <- lda(y ~ x, data = train)
lda.pred <- predict(lda.fit, test)$class
mean(lda.pred != test$y)

qda.fit <- qda(y ~ x, data = train)
qda.pred <- predict(qda.fit, test)$class
mean(qda.pred != test$y)
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

#' Problem 5

#head(Boston)

#' 1.
set.seed(1)
poly.fit <- glm(nox ~ poly(dis, 3), data = Boston)
summary(poly.fit)
dislims <- range(Boston$dis)
dis.grid <- seq(from = dislims[1], to = dislims[2], by = .1)
pred.value <- predict(poly.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "grey")
lines(dis.grid, pred.value)


#' 2.

rss <- rep(0, 10)
for (i in 1:10) {
  poly.fit <- lm(nox ~ poly(dis, i), data = Boston)
  rss[i] <- sum(poly.fit$residuals^2)
}

rss[c(1,3,5,7,10)]

plot(c(1,3,5,7,10),rss[c(1,3,5,7,10)], xlab = "degree", ylab = "RSS", type = "l")


#' 3.

set.seed(1)
#install.packages("ISLR")
library(ISLR)
library(boot)
cv.error <- rep(0,10)
for (i in 1:10) {
  poly.fit <- glm(nox ~ poly(dis, i), data = Boston)
  cv.error[i] <- cv.glm(Boston, poly.fit, K=10)$delta[1]
}

plot(c(1,3,5,7,10),cv.error[c(1,3,5,7,10)],xlab = "degree",type = "l")
#index not value
which.min(cv.error[c(1,3,5,7,10)])

#' Because the cv error is the lowest when the polynomial degree is 3, we should
#' choose degree 3 (for this seed).


#' 4.

library(splines)
bs.fit <- glm(nox ~ bs(dis, knots = c(6, 10)), data = Boston)
summary(bs.fit)

#' From the plot in (1), we see that are probably some change in trend at 
#' dis = 6 and dis = 10, thus, I choose these two numbers as knots.

#' The resulting fit is:

pred <- predict(bs.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "grey")
lines(dis.grid, pred)


#' 5.

rss <- rep(0, 10)
for (i in 4:10) {
  bs.fit <- glm(nox ~ bs(dis, df = i), data = Boston)
  rss[i] <- sum(bs.fit$residuals^2)
}
rss[c(4,6,8,10)]

plot(c(4,6,8,10),rss[c(4,6,8,10)],xlab = "df", ylab = "RSS", type = "l" )

#' We see that the RSS keeps dropping as the degree of freedom increases when we
#' focus on df = 4,6,8,10.



#' 6.
options(warn=-1)
set.seed(1)
cv.error <- c()
for (i in 4:10) {
  bs.fit <- glm(nox ~ bs(dis, df = i), data = Boston)
  cv.error[i] <- cv.glm(Boston, bs.fit, K = 10)$delta[1]
}

plot(c(4,6,8,10),cv.error[c(4,6,8,10)],xlab = "df", type = "l")

#index not value
which.min(cv.error[c(4,6,8,10)])

#' I start from df = 4 because it will be too small if it's less than 3.
#' For this seed, the cv error achieve its minimum at df = 10 (index = 4), and 
#' thus, we may choose df = 10 for the basic spline regression.

