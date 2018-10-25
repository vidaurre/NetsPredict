tmpnm <- commandArgs(trailingOnly = TRUE)

library(glmnet)
source('call_setpars.R')
N = dim(X)[1]; p  = dim(X)[2]

dev = rep(0,p)

for (j in 1:p) {
x = cbind(X[,j],matrix(0,N,1))
glmfit = glmnet(x,Y,family=family, alpha = parameters$alpha,
    nlambda = parameters$nlambda, lambda=parameters$lambda,
    standardize = parameters$standardize, intercept=parameters$intercept)
dev[j] = glmfit$dev[length(glmfit$dev)]
}

write.table(dev, paste(tmpnm,'/out/dev',sep=''), row.names = FALSE, col.names = FALSE)
