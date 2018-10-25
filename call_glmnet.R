tmpnm <- commandArgs(trailingOnly = TRUE)

library(glmnet)
source('call_setpars.R')
N = dim(X)[1]; p  = dim(X)[2]
if (p==1) X = cbind(X,matrix(0,N,1))

if (is.null(parameters$lambda)) {
glmfit = glmnet(X,Y,family=family, alpha = parameters$alpha,
    nlambda = parameters$nlambda,
    standardize = parameters$standardize, intercept=parameters$intercept)
} else {
glmfit = glmnet(X,Y,family=family, alpha = parameters$alpha,
    lambda=parameters$lambda[,1],
    standardize = parameters$standardize, intercept=parameters$intercept)
}

if (family=="multinomial") {
q = dim(Y)[2]
for (j in 1:q) {
    write.table(as.matrix(glmfit$beta[[j]]), paste(tmpnm,'/out/beta',j,sep=''), row.names = FALSE, col.names = FALSE)
}
} else {
write.table(as.matrix(glmfit$beta), paste(tmpnm,'/out/beta',sep=''), row.names = FALSE, col.names = FALSE)
}

write.table(glmfit$a0, paste(tmpnm,'/out/a0',sep=''), row.names = FALSE, col.names = FALSE)
write.table(glmfit$lambda, paste(tmpnm,'/out/lambda',sep=''), row.names = FALSE, col.names = FALSE)
#write.table(glmfit$df, paste(tmpnm,'/out/df',sep=''), row.names = FALSE, col.names = FALSE)
write.table(glmfit$offset+0, paste(tmpnm,'/out/offset',sep=''), row.names = FALSE, col.names = FALSE)
