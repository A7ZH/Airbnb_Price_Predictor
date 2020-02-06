library(MASS)
library(tidyverse)
library(loon.ggplot)
library(maps)
library(loon)
library(rgeos)
library(maptools)
library(geojsonio)
library(ggplot2)
library(gridExtra)
library(GGally)
library(sentimentr)
library(rasterly)
library(reshape2)
library(mgcv)
library(stringr)
library(purrr)
library(rvest)
data <- read_csv("AB_NYC_2019.csv")
colnames(data)

data <- data %>% 
  dplyr::select(-c("id", "host_id", "host_name", "neighbourhood", "last_review")) %>% 
  na.omit()
data
# target price
#### price is zero?
n <- dim(data)[1]
zero_id <- which(data$price == 0)
valid_id <- setdiff(1:n, c(zero_id))
#### text analysis

comment <- data$name %>% 
  get_sentences() %>% 
  sentiment()
comment <- comment[, sum(sentiment), by = element_id]
colnames(comment) <- c("element_id", "sentiment_score")
data <- data %>% 
  mutate(sentiment_score = comment$sentiment_score) %>% 
  dplyr::select(-name)

set.seed(12345)
n <- length(valid_id)
test_id <- sample(1:n, ceiling(n/4))
train_id <- setdiff(1:n, test_id)
data_test <- data[valid_id[test_id], ]
data_train <- data[valid_id[train_id], ]
n_train <- dim(data_train)[1]

######################## Maps ########################

# this is the geojson of the NYC community districts
URL <- "http://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/nycd/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=geojson"
fil <- "nyc_community_districts.geojson"
if (!file.exists(fil)) download.file(URL, fil)

nyc_districts <- geojson_read(fil, what="sp")
nyc_districts_map <- fortify(nyc_districts, 
                             region="BoroCD")

color_neighbor_labels <- loon::l_getColorList()[1:6][-1]
neighbor_labels <- c("Manhattan", "The Bronx", "Brooklyn", "Queens", "Staten Island")
colorize <- function(x) {
  sub_str <- substr(x, 1, 1)
  color_neighbor_labels[as.numeric(sub_str)]
}

# NewYork Map
category_price <- c("economical", "mediocre", "expensive", "luxury")
hist(log(data$price))
# "0~4", "4 ~ 5", "5 ~ 6", "6 ~"
cate_price <- vapply(data$price,
                     function(p) {
                       if(log(p) <= 4) {
                         category_price[1]
                       } else if(log(p) <= 5) {
                         category_price[2]
                       } else if(log(p) <= 6) {
                         category_price[3]
                       } else {
                         category_price[4]
                       }
                     }, character(1))

map_fill <- data.frame(district=nyc_districts@data$BoroCD,
                       fill=sapply(nyc_districts@data$BoroCD, colorize))

gg <- ggplot(data = data %>% mutate(cate_price = cate_price)) + 
  geom_map(data=nyc_districts_map, map = nyc_districts_map,
           aes(x=long, y=lat, map_id = id), 
           size=0.15, fill=NA) + 
  geom_map(data=map_fill, map=nyc_districts_map,
           aes(fill=fill, map_id=district),
           size=0.15, color = "#2b2b2b",
           alpha = 0.6) + 
  geom_point(
    mapping = aes(x = longitude, y = latitude, 
                  color = cate_price),
    alpha = 0.5
  ) + 
  scale_fill_identity(labels = setNames(c(neighbor_labels, "Landmark"), 
                                        c(color_neighbor_labels, "magenta")),
                      guide = "legend")
gg
# landmark
landmark <- data.frame(
  name = c('One World Trade Center', 
           'Times Square', 
           'Madison Square', 
           'Barclays Center', 
           'Statue of Liberty',
           'Empire State Building'),
  lat = c(40.712742, 40.758896, 40.750504, 40.682661, 40.689247, 40.748817),
  long = c( -74.013382, -73.985130,-73.993439, -73.975225, -74.044502, -73.985428)
)

pn <- function(x) {
  len <- length(x)
  id <- (1:len) %% 2 == 0
  x[id] <- -x[id]
  x
}
theta <- pi/4
l <- dim(landmark)[1]
line_coord <-  pn(0.1 + seq(from = 0, to = 0.1, length.out = l))
x_line <- c(landmark$long - cos(theta) * line_coord, 
            landmark$long - pn(rep(1e-2, l)))
y_line <- c(landmark$lat + sin(theta) * abs(line_coord), 
            landmark$lat +  1e-2)

line_data <- data.frame(x = x_line,
                        y = y_line,
                        group = rep(1:l, 2))
line_data1 <- line_data %>% 
  filter(group %in% c(seq(1, by = 2, len = ceiling(l/2))))
line_data2 <- line_data %>% 
  filter(group %in% c(seq(2, by = 2, len = ceiling(l/2))))

gg + 
  geom_polygonGlyph(
    data = landmark,
    mapping = aes(x = long, y = lat),
    fill = "magenta",
    size = 2,
    polygon_x = lapply(1:l, function(i) x_star),
    polygon_y = lapply(1:l, function(i) y_star)
  ) + 
  geom_line(
    data = line_data1,
    mapping = aes(x, y, group = group),
    arrow = arrow(length=unit(0.30,"cm"), ends = "last", type = "closed")
  ) + 
  geom_line(
    data = line_data2,
    mapping = aes(x, y, group = group),
    arrow = arrow(length=unit(0.30,"cm"), ends = "first", type = "closed")
  ) + 
  geom_text(
    data = data.frame(x = landmark$long - cos(theta) * line_coord - pn(rep(1e-2, l)),
                      y = landmark$lat + sin(theta) * abs(line_coord) +  1e-2,
                      text = landmark$name),
    mapping = aes(x, y, label = text)
  )
# price
gg + facet_wrap(~neighbourhood_group)
# room type
gg + facet_wrap(~room_type)
# neighbourhood_group and room_type
gg + facet_grid(neighbourhood_group~room_type)

# price heatmap
rasterly(data  = data_train,
         mapping = aes(x = longitude, y = latitude, on = price)) %>% 
rasterly_points(color = viridis_map, 
                reduction_func = "mean",
                background = "black") -> ras
ras
image2data(ras) %>% 
  filter(color != "black") -> imageData
ggplot() + 
  geom_map(data=nyc_districts_map, map = nyc_districts_map,
           aes(x=long, y=lat, map_id = id), 
           size=0.15, fill=NA) + 
  geom_map(data=map_fill, map=nyc_districts_map,
           aes(fill=fill, map_id=district),
           size=0.15, color = "#2b2b2b",
           alpha = 0.6) + 
  geom_point(
    data = imageData,
    mapping = aes(x = x, y = y),
    color = imageData$color,
    shape = 15,
  ) + 
  scale_fill_identity(labels = setNames(neighbor_labels, color_neighbor_labels),
                      guide = "legend") + 
  geom_polygonGlyph(
    data = landmark,
    mapping = aes(x = long, y = lat),
    fill = "magenta",
    size = 2,
    polygon_x = lapply(1:l, function(i) x_star),
    polygon_y = lapply(1:l, function(i) y_star)
  ) + 
  geom_line(
    data = line_data1,
    mapping = aes(x, y, group = group),
    arrow = arrow(length=unit(0.30,"cm"), ends = "last", type = "closed")
  ) + 
  geom_line(
    data = line_data2,
    mapping = aes(x, y, group = group),
    arrow = arrow(length=unit(0.30,"cm"), ends = "first", type = "closed")
  ) + 
  geom_text(
    data = data.frame(x = landmark$long - cos(theta) * line_coord - pn(rep(1e-2, l)),
                      y = landmark$lat + sin(theta) * abs(line_coord) +  1e-2,
                      text = landmark$name),
    mapping = aes(x, y, label = text),
    color = "red"
  )
######################## relationship with price ########################
# latitude
data_train %>% 
  ggplot(mapping = aes(x = latitude, 
                       y = log(price))) + 
  geom_point(mapping = aes(color = neighbourhood_group)) + 
  geom_smooth(method = "lm")
# longitude
data_train %>% 
  ggplot(mapping = aes(x = longitude, 
                       y = log(price))) + 
  geom_point(mapping = aes(color = neighbourhood_group)) + 
  geom_smooth(method = "lm")
# reviews_per_month
data_train %>% 
  ggplot(mapping = aes(x = log(reviews_per_month), 
                       y = log(price))) + 
  geom_point(mapping = aes(color = neighbourhood_group)) + 
  geom_smooth(method = "lm")

# sentiment_score
data_train %>% 
  ggplot(mapping = aes(x = sentiment_score, 
                       y = log(price)))  + 
  geom_point(mapping = aes(color = neighbourhood_group)) + 
  geom_smooth(method = "lm")

######################## price ########################
data_train %>% 
  ggplot(mapping = aes(x = neighbourhood_group, 
                       y = log(price), 
                       fill = neighbourhood_group)) + 
  geom_boxplot() + 
  stat_summary(fun.y=mean, geom="line", size=2, color="magenta", aes(group=1)) -> price_neighbour
price_neighbour

data_train %>% 
  ggplot(mapping = aes(y = log(price), fill = room_type)) + 
  geom_boxplot() -> price_room
price_room
######################## minimum_nights and room_type ########################
# entire home/apt requires the largest minimum stay.
data_train %>% 
  ggplot(mapping = aes(y = minimum_nights, fill = room_type)) + 
  geom_boxplot(outlier.shape = NA) +
  scale_y_continuous(limits = quantile(data_train$minimum_nights, c(0.1, 0.9))) + 
  coord_cartesian(ylim = c(0, 7.5)) + 
  ggtitle("Outliers removed boxplot")

######################## number_of_reviews ########################
# seems like shared room has less attractions
data_train %>% 
  ggplot(mapping = aes(y = log(number_of_reviews), fill = room_type)) + 
  geom_boxplot()

data_train %>% 
  dplyr::select(neighbourhood_group, number_of_reviews) %>%
  ggplot(mapping = aes(x = neighbourhood_group, 
                       y = log(number_of_reviews), 
                       fill = neighbourhood_group)) + 
  geom_boxplot() +
  stat_summary(fun.y=mean, geom="line", size=2, color="magenta", aes(group=1)) ->
  reviews
  
gridExtra::grid.arrange(grobs = list(price_neighbour + 
                                       theme(legend.position = "none"), 
                                     reviews), 
                        layout_matrix = matrix(c(1,1,2,2,2), nrow = 1))

######################## availability_365 ########################
data_train %>% 
  ggplot(mapping = aes(y = availability_365, fill = room_type)) + 
  geom_boxplot() + 
  facet_grid(~neighbourhood_group)

data_train %>% 
  ggplot(mapping = aes(y = availability_365, fill = neighbourhood_group)) + 
  geom_boxplot()+ 
  facet_grid(~room_type)

######################## sentiment_score ########################
data_train %>% 
  ggplot(mapping = aes(y = sentiment_score, 
                       fill = room_type)) + 
  geom_boxplot() + 
  facet_grid(~neighbourhood_group)

data_train %>% 
  ggplot(mapping = aes(y = sentiment_score, fill = neighbourhood_group)) + 
  geom_boxplot() + 
  facet_grid(~room_type)

######################## models ########################

# GCV
gcv <- function(D, df, n) {
  n * D /(n - df)^2
}

## linear model
# data_train <- data_train %>% dplyr::select(-neighbourhood_group)
fitAll <- lm(log(price)~., data = data_train)
fit <-  stepAIC(
  lm(log(price) ~ 1, data = data_train),
  scope = formula(fitAll),
  direction = "forward",
  k = log(n),
  # set trace to TRUE can visualize the steps of selection
  trace = FALSE
)
# Add some intersections terms
fit0 <- lm(log(price) ~ (room_type) * (longitude + availability_365 + neighbourhood_group + 
                                         latitude + minimum_nights + number_of_reviews + sentiment_score), 
           data = data_train)
# Any commend?
anova(fit, fit0)
fit1 <- lm(log(price) ~ (neighbourhood_group) * (longitude + availability_365 + room_type + 
                                                   latitude + minimum_nights + number_of_reviews + sentiment_score), 
           data = data_train)
# Any commend?
anova(fit, fit1)
# fit1 beats fit0
anova(fit0, fit1)
fit2 <- lm(log(price) ~ (room_type + neighbourhood_group) * (longitude + availability_365 + 
                                                               latitude + minimum_nights + number_of_reviews + sentiment_score), 
           data = data_train)
anova(fit1, fit2)
anova(fit0, fit2)

fit3 <- lm(log(price) ~ (room_type * neighbourhood_group) * (longitude + availability_365 + 
                                                               latitude + minimum_nights + number_of_reviews + sentiment_score), 
           data = data_train)
anova(fit2, fit3)

library(mgcv)
data_train <- data_train %>% 
  mutate(new_factor = paste(room_type,  neighbourhood_group, sep = ":"))
data_train$new_factor <- factor(data_train$new_factor)

data_test <- data_test %>% 
  mutate(new_factor = paste(room_type,  neighbourhood_group, sep = ":"))
data_test$new_factor <- factor(data_test$new_factor)
# Basic Smoothing (15 ~ 20 mins)
# fits <- gam(log(price) ~ (s(longitude, by = new_factor) + s(availability_365, by = new_factor) +
#                                 s(latitude, by = new_factor) + s(minimum_nights, by = new_factor) +
#                                 s(number_of_reviews, by = new_factor) + s(sentiment_score, by = new_factor)),
#                 data = data_train)
# saveRDS("smoothing.rds")
# fits <- readRDS("smoothing.rds")
# anova(fit3, fits)


# Tensor product smooths
# Aim is to construct a basis and penalties that are
# invariant to rescaling of any or all of the covariates

## Cubic
fitCubic <- gam(log(price) ~ (te(longitude, by = new_factor) + te(availability_365, by = new_factor) +
                            te(latitude, by = new_factor) + te(minimum_nights, by = new_factor) +
                            te(number_of_reviews, by = new_factor) + te(sentiment_score, by = new_factor)),
            data = data_train)
anova(fit3, fitCubic)
# Thin plate
fitThinPlate <- gam(log(price) ~ (te(longitude, by = new_factor, bs="tp") + 
                                    te(availability_365, by = new_factor, bs="tp") +
                                te(latitude, by = new_factor, bs="tp") + 
                                  te(minimum_nights, by = new_factor, bs="tp") +
                                te(number_of_reviews, by = new_factor, bs="tp") + 
                                  te(sentiment_score, by = new_factor, bs="tp")),
                data = data_train)
anova(fit3, fitThinPlate)
anova(fitThinPlate, fitCubic)
# cubic is better!
pred_gam <- predict(fitCubic, newdata = data_test %>%
                  dplyr::select(longitude, new_factor, availability_365,
                                latitude, minimum_nights, number_of_reviews, sentiment_score))

gcv(sum(pred_gam - log(data_test$price))^2, 
    df = n_train - fitCubic$df.residual, 
    n = n_train)

gcv(sum(pred_gam - log(data_test$price))^2, 
    df = 0, 
    n = n_train)

write.csv(data_train, "data_train.csv")
write.csv(data_test, "data_test.csv")
######################## XGboost ########################

var2num <- function(x, colnames = NULL) {
  uni_x <- unique(x)
  n_uni_x <- length(uni_x)
  n <- length(x)
  lapply(1:n, 
         function(i) {
           num_x <- rep(0, n_uni_x - 1)
           id <- which(x[i] == uni_x)
           if(id < n_uni_x) {
             num_x[id] <- 1
           }
           num_x
         }) %>% 
    unlist() %>% 
    matrix(ncol = n_uni_x - 1,  byrow = T) %>%
    as_tibble() -> tib
  if(!is.null(colnames)) 
    colnames(tib) <- colnames
  tib
}

library(xgboost)
dat_train_ <- data_train %>% 
  dplyr::select(-c(price, new_factor))
dat_test_ <- data_test %>%
  dplyr::select(-c(price, new_factor))
dat_train_ <- dat_train_ %>% 
  cbind(var2num(dat_train_$room_type)) %>% 
  cbind(var2num(dat_train_$neighbourhood_group, 
                colnames = paste0("V", 5:8))) %>%
  select(-c(room_type, neighbourhood_group)) %>% 
  as_tibble()
dat_test_ <- dat_test_ %>% 
  cbind(var2num(dat_test_$room_type)) %>% 
  cbind(var2num(dat_test_$neighbourhood_group, 
                colnames = paste0("V", 5:8))) %>%
  select(-c(room_type, neighbourhood_group)) %>% 
  as_tibble()
dtrain <- xgb.DMatrix(data = as.matrix(dat_train_), label = log(data_train$price))
Xbst <- xgboost(data = dtrain, nrounds = 10,
                verbose = FALSE)
pred_xgboost <- predict(Xbst, 
                        xgb.DMatrix(data = as.matrix(dat_test_)))
gcv(sum(pred_xgboost - log(data_test$price))^2, 
    df = 0, 
    n = n_train)
######################## KNN ########################
library(class)
pred_knn <- knn(dat_train_, dat_test_, log(data_train$price), 
                k = 3, prob=FALSE) %>% 
  as.character() %>% 
  as.numeric()
gcv(sum(pred_knn - log(data_test$price))^2, 
    df = 0, 
    n = n_train)

######################## randomForest ########################
library(randomForest)
forest <- randomForest(x = dat_train_, 
                       y = log(data_train$price), 
                       ntree = 20) 

pred_forest <- predict(forest, dat_test_)
gcv(sum(pred_forest - log(data_test$price))^2, 
    df = 0, 
    n = n_train)
######################## DNN ########################
library(keras)
train_data <- as.matrix(dat_train_) %>% 
  scale()
y_train <- data_train$price %>% log()
y_test <- data_test$price %>% log()

# Use means and standard deviations from training set to normalize test set
test_data <- as.matrix(dat_test_) %>% 
  scale(center = attr(train_data, "scaled:center") , 
        scale = attr(train_data, "scaled:scale"))

build_model <- function(layer1 = 128, 
                        layer2 = 32, 
                        layer3 = 3, train = train_data) {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = layer1, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = layer2, activation = "relu") %>%
    layer_dense(units = layer3)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model() 

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 500

# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)

test_predictions <- model %>% 
  predict(test_data) %>% 
  c()
gcv(sum(test_predictions - y_test)^2, 
    df = 0, 
    n = n_train)
