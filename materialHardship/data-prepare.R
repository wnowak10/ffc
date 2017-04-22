## Prepare data just for materialHardship

library(data.table)
library(tidyverse)

#Not using randomness here, but whatever
set.seed(1234)

#This assumes you're in the working directory with the files.
print("Reading in CSV files. Why doesn't Will capitalize words?")

background <- fread("background.csv")
prediction <- fread("prediction.csv")
train      <- fread("train.csv")

print("Files read. Why doesn't Will punctuate sentences?")

# Background is super WIDE. 
# We have only 4242 IDs, but almost 13k features!
# Yikes!!

dim(background); dim(prediction); dim(train)

#OK...so they want predictions for...everything.
# So we are going to make a **training** set,
# using the rows of background that
# have non-NA values for materialHardship. Fine.
# 

train.no.na.mh <- train[(!is.na(materialHardship))]

# Then we need to make predictions for every challengeID that is
# (a) not in this training set, or
# (b) is in the training set, but has NA for materialHardship 

prediction.ids <- unique(c( train[is.na(materialHardship),challengeID],
                     background[,setdiff(challengeID,
                                         train.no.na.mh[,challengeID])]))

stopifnot( length(prediction.ids) + nrow(train.no.na.mh) == nrow(background) )

#OK. Now pull these ideas from the background set

#keying and joining are important data.table concepts.
#This basically means "tell background to organize itself so that
# challengeID is the identifying variable for a row,
setkey(background,challengeID)
# then look up all the challengeIDs that match prediction.ids
# and pull those records.
background.subset.for.prediction <- background[J(prediction.ids)]
#See? Remark that there as many rows as length(prediction.ids)
dim(background.subset.for.prediction)
length(prediction.ids)

#Now let's do the same with the IDs we *are* going to train on
#Here, 
train.ids <- train.no.na.mh[,challengeID]
background.subset.for.training <- background[J(train.ids)]

#So now we have two sets of data, one for training the model,
#and one for out-of-sample predictions. 
#Notice that the row dimensions add up:

dim(background.subset.for.training)
dim(background.subset.for.prediction)

stopifnot(
    nrow(background.subset.for.training) + 
    nrow(background.subset.for.prediction)  ==
    nrow(background) 
)

#Now write these suckers out for easy use later.

fwrite(background.subset.for.training,file="background-train_only.csv")
fwrite(background.subset.for.prediction,file="background-predict_only.csv")
