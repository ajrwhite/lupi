# KNOWLEDGE TRANSFER
# Ported from Python script by Paolo Toccaceli

# LIBRARIES
# I have attempted to use base R functions wherever possible
# 1. EBImage is used in resizeImages; next two lines are to install it in your R environment
# source("http://bioconductor.org/biocLite.R")
# biocLite("EBImage")
library(EBImage) # EBImage::resize() is called in resizeImages
# 2. CVST - "Fast Cross-Validation via Sequential Testing"
# install.packages("CVST")
library(CVST) # Used to emulate SKLearn CV, SVM and KRR functionality
# 3. xgboost - "eXtreme Gradient Boosting"
# install.packages("xgboost")
library(xgboost)

# KEY VARIABLES
setwd("C:/Users/Andrew.000/Desktop/MSc Data Science and Analytics/CS5821-2 FINAL PROJECT")
train_image_file = ("MNIST\\t10k-images-idx3-ubyte")
train_label_file = ("MNIST\\t10k-labels-idx1-ubyte")

# SCRATCHBOOK FOR TESTING CODE
train_images = readDat(train_image_file, 10000)
train_images_resize = resizeImages(train_images, 28, 28, 16, 16)
train_labels = readLabels(train_label_file, 10000)
# calculate skew on 1000 images as privileged information
priv_info_skew = rep(0, 1000)
priv_info_holes = rep(0, 1000)
for (i in 1:1000) {
  temp_image = extractImage(train_images, 28, 28, i)
  priv_info_skew[i] = calcSkew(temp_image)
  priv_info_holes[i] = calcHoles(temp_image)
}
# Develop our phi model for calculating skew on remaining images
phi_model_skew = xgb.cv(data=train_images[1:1000,],
                   label=priv_info_skew,
                   nrounds=100,
                   print=10,
                   max_depth=100,
                   eta=0.1,
                   nfold=4,
                   nthread=3)
# Develop our phi model for calculating holes on remaining images
phi_model_holes = xgboost(data=train_images[1:1000,],
                         label=priv_info_holes,
                         nrounds=500,
                         print=10,
                         max_depth=30,
                         eta=0.01,
                         nfold=4,
                         nthread=3,
                         objective="multi:softmax",
                         num_class=4)
# Add phi features to remaining data
pred_priv_info_skew = predict(phi_model_skew, train_images[1001:10000,])
pred_priv_info_holes = predict(phi_model_holes, train_images[1001:10000,])
train_and_phi = cbind(train_images, c(priv_info, pred_priv_info))
# Build model on X and X*
xgbcv = xgb.cv(data=train_and_phi,
               label=train_labels, 
               nrounds=100,
               nfold=4,
               objective="multi:softmax",
               num_class=10,
               print=10,
               max_depth=20,
               eta=0.1,
               nthread=3)
# Build model just on X
xgbcv = xgb.cv(data=train_images,
               label=train_labels, 
               nrounds=100,
               nfold=4,
               objective="multi:softmax",
               num_class=10,
               print=10,
               max_depth=20,
               eta=0.1,
               nthread=3)

# FUNCTIONS

# Function: readDat - Read in the MNIST files as a matrix [COMPLETE] ---------------------------
readDat = function(filename, n) {
  infile = file(filename, "rb") # open image file connection
  on.exit(close(infile)) # close connection when function exits
  readBin(infile, integer(), n=4, endian="big")
  outmat = matrix(as.double(readBin(infile, integer(), size=1, n=28*28*n, endian="big", signed=F)), nrow=n, ncol=28*28, byrow=T)
  return(outmat)
}

# Function: readLabels - Read in the MNIST labels as a vector [COMPLETE] --------
readLabels = function(filename, n) {
  infile = file(filename, "rb") # open label file connection
  on.exit(close(infile)) # close connection when function exits
  readBin(infile, integer(), n=2, endian="big")
  return(readBin(infile, integer(), size=1, n=n, endian="big"))
}

# Function: resizeImages - Resize the images [COMPLETE BUT SLOW] ---------------------------------------------
resizeImages = function(images, inx, iny, outx, outy) {
  n = nrow(images)
  outmat = rep(0, outx*outy)
  for (i in 1:n) {
    temp = as.vector(resize(matrix(images[i,], inx, iny), outx, outy))
    outmat = rbind(outmat, temp)
  }
  return(outmat[-1,])
}

# Function: showDigit - Display a digit [COMPLETE] ------------------------
showDigit = function(single_image) {
  return(image(single_image, col=grey(seq(1, 0, length=256))))
}


# Function: extractImage - Extract image from vector [COMPLETE] -----------
extractImage = function(image_matrix, w, h, item) {
  return(matrix(image_matrix[item,], w, h)[,h:1])
}

# Function: calcSkew - calculate skewness of image [COMPLETE] ------------------------
calcSkew = function(single_image) {
  # Extract each non-white pixel as an x, y point
  extracted_coords = which(single_image > 0, arr.ind=T)
  return(cor(extracted_coords[,1], extracted_coords[,2]))
}

# Function: calcHoles - calculate number of holes in image [COMPLETE BUT SLOW] ----------------
calcHoles = function(single_image) {
  single_image = (single_image > 0) * 1
  single_image = floodFill(single_image)
  hole_cells = which(single_image == 0, arr.ind=T)
  if(nrow(hole_cells) < 2) {
    num_holes = nrow(hole_cells)
  } else if (nrow(hole_cells) == 2) {
    num_holes = ifelse(max(abs(hole_cells[1,] - hole_cells[2,])) > 1, 2, 1)
  } else {
    hole_cluster = cutree(hclust(dist(hole_cells, "maximum"), "single"), h = 1)
    num_holes = length(unique(hole_cluster))
  }
  return(num_holes)
}

# Function: floodFill - flood fill an image from the boundary [COMPLETE] -------------
floodFill = function(single_image) {
  w = ncol(single_image)
  h = nrow(single_image)
  # FLOOD FILL
  # fill boundaries
  single_image[1,single_image[1,]==0] = -1
  single_image[single_image[,1]==0,1] = -1
  single_image[h,single_image[h,]==0] = -1
  single_image[single_image[,w]==0,w] = -1
  # Repeat each pass twice to ensure spirals are filled
  for (r in 1:2) {
  # pass top to bottom
    for (i in 2:(h-1)) {
      for (j in 2:(w-1)) {
        if((single_image[i-1,j]==-1 |
            single_image[i-1,j-1]==-1 |
            single_image[i-1,j+1]==-1) & single_image[i,j]==0) single_image[i,j] = -1
      }
    }
    # pass bottom to top
    for (i in (h-1):2) {
      for (j in (w-1):2) {
        if((single_image[i+1,j]==-1 |
            single_image[i+1,j-1]==-1 |
            single_image[i+1,j+1]==-1) & single_image[i,j]==0) single_image[i,j] = -1
      }
    }
    # pass left to right
    for (j in 2:(w-1)) {
      for (i in 2:(h-1)) {
        if((single_image[i,j-1]==-1 |
            single_image[i-1,j-1]==-1 |
            single_image[i+1,j-1]==-1) & single_image[i,j]==0) single_image[i,j] = -1
      }
    }
    # pass right to left
    for (j in (w-1):2) {
      for (i in (h-1):2) {
        if((single_image[i,j+1]==-1 |
            single_image[i-1,j+1]==-1 |
            single_image[i+1,j+1]==-1) & single_image[i,j]==0) single_image[i,j] = -1
      }
    }
  }
  return(single_image)
}

# Function: crossValidatedSVC - Perform cross-validated support vector classification ---------
crossValidatedSVC = function(trainX,
                             trainY,
                             param_grid,
                             plotTitle=NULL,
                             plotFileName=NULL,
                             kernel='rbf',
                             verbose=False) {
  return()
}

# Function: 

# Function: validatedSVC - Perform validated support vector classification --------
validatedSVC = function(trainX,
                        trainY,
                        validationX,
                        validationY,
                        param_grid,
                        plotTitle=NULL,
                        plotFileName=NULL,
                        kernel='rbf',
                        verbose=False) {
  return()
}

# Function: logRange - output a log range for parameter testing [COMPLETE] -----------
logRange = function(rmin,rmax,valPerUnit,base) {
  return(base^seq(rmin, rmax, 1/valPerUnit))
}

# Function: inkKernel - INK-Spline kernel of degree 1 [OPTIONAL] ---------------------
inkKernel = function(p, q) {
  return()
}

# Function: inkKernelNorm - calculate norm of INK-Spline kernel [OPTIONAL] -----------
inkKernelNorm = function(p, q) {
  return()
}

# Function: quadKernel - calculate quadratic kernel [COMPLETE] -----------------------
quadKernel = function(xi, xj) {
  return((xi %*% xj)^2)
}

# Function: newDecisionFunction - calculate SVM decision function [OPTIONAL] --------
newDecisionFunction = function(u, b, x, intercept) {
  return()
}

# Function: approx_f - calculate approx decision function in terms of phis [CHECK!] --------
approx_f = function(phi,beta,intercept) {
  return(phi %*% beta[1:nrow(phi)] + intercept) # Not sure if phi is a vector or a matrix
}

# Function: phiKRRWithVal - phi kernel ridge regression with validation --------
phiKRRWithVal = function(trainX,
                         trainY,
                         yLabels,
                         param_grid,
                         args,
                         kwargs) {
  cvstData = constructData(x=trainX, y=trainY)
  clf = constructKRRLearner()
  clf_fit = clf$learn(cvstData, bestparams)
}
