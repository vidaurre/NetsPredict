X = as.matrix(read.table(paste(tmpnm,'/X.dat',sep='')))
Y = as.matrix(read.table(paste(tmpnm,'/Y.dat',sep='')))
family <- readLines(paste(tmpnm,'/family.dat',sep=''),encoding="UTF-8")
fparameters = list.files(paste(tmpnm,'/params',sep=''))

p  = max(dim(X)[2],2)
parameters = list()
parameters$alpha = 1
parameters$lambda = NULL
parameters$intercept = 1
parameters$standardize = 1
#parameters$dfmax = p + 1
#parameters$pmax = min(parameters$dfmax * 2+20, p)
parameters$nlambda = 100

for (par in fparameters) {
val = read.table(paste(tmpnm,'/params/',par,sep=''))
parameters[[par]] = val
}

# for precision issues
if (family=='multinomial') {
sumY = apply(Y,1,sum)
for (j in 1:dim(Y)[2]) Y[,j] = Y[,j] / sumY;
}
