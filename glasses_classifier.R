library(e1071)
library(biOps)

# 作業ディレクトリの指定
kBaseDir <-"./" 

# 縮小サイズ
kXBase = 48
kYBase = 27
kXYVecMax = kXBase * kYBase

# グリッドサーチ用変数
# 大雑把な探索
kRoughGammaRange <-c(2^(seq(-15,3,3)))
kRoughCostRange <-c(2^(seq(-5,15,3)))
# 細かな探索
kDetailGammaRange <-c(2^(seq(-16,-14,0.5)))
kDetailCostRange <-c(2^(seq(5,7,0.5)))

CreateDataset <- function(){
  # 画像オブジェクト生成
  gl_imgs <- ReadImages("../data/glasses/")
  none_imgs <- ReadImages("../data/no_glasses/")
  # 素性ベクトル作成
  gl_df <- ConvertImagesToFeatureVectors(gl_imgs, "glasses")
  none_df <- ConvertImagesToFeatureVectors(none_imgs, "none")
  # データセット作成
  df <- rbind(gl_df, none_df)
  return(df)  
}

CreateSamples <- function(type){
  if(type == "glasses"){
    gl_imgs <- ReadImages("../data/sample_glasses/")
    df <- ConvertImagesToFeatureVectors(gl_imgs, "glasses")
    
  } else {
    none_imgs <- ReadImages("../data/sample_no_glasses/")
    df <- ConvertImagesToFeatureVectors(none_imgs, "none")
  }
  df$label <- NULL
  return(df)  
}

ReadImages <- function(dir){
  filenames <- list.files(dir, "*.jpg", full.names=T)
  images <- lapply(filenames, readJpeg)
  return(images)
}

# 読み込んだ jpg を kXBase x kYBase にダウンサイジング
DoDownsising <- function(img){
  x_size <- ncol(img)
  y_size <- nrow(img)
  x_scale <- kXBase / x_size
  y_scale <- kYBase / y_size
  img <- imgAverageShrink(img, x=x_scale, y=y_scale)
  return(img)
}


# 読み込んだ jpg を グレースケール変換
ToGrayscale <- function(img){
  return(imgRGB2Grey(img, coefs=c(0.30, 0.59, 0.11)))
}

# 読み込んだ jpg を エッジ強調
EmphasizeEdge <- function(img){
  img <- imgCanny(img, sigma=0.4)
  return(img)
}

# 素性ベクトル作成
ConvertImageToFeatureVector <- function(img){
  # 1. 元画像
  # plot(img)
  # browser()
  # 2. グレースケール変換
  # black: 0, white: 255
  g_img <- ToGrayscale(img)
  # plot(g_img)
  # browser()  
  # 3. エッジ強調
  ge_img <- EmphasizeEdge(g_img)
  # plot(ge_img)
  # browser()  
  # 4. ダウンサイジング
  ged_img <- DoDownsising(ge_img)
  # plot(ged_img)
  # browser()  
  vec <- as.vector(ged_img)
  # black: 0 to white: 1 
  normalized_vec <- vec / 255
  return(normalized_vec)
}

# 全イメージを素性ベクトルに変換
ConvertImagesToFeatureVectors <- function(imgs, label){
  vectors_list <- lapply(imgs, ConvertImageToFeatureVector)
  vectors_list <- lapply(vectors_list, (function(x) {x[1:kXYVecMax]}))
  df <- as.data.frame(do.call("rbind", vectors_list))
  df[is.na(df)] <- 1
  df$label <- as.factor(label)
  return(df)
}

# パラメータチューニング
TuneSVM <- function(dataset, gammas, costs){
  t <- tune.svm(
    label ~ ., 
    data = dataset,
    gamma = gammas,
    cost = costs,
    tunecontrol = tune.control(sampling="cross", nrow(dataset))
  )
  cat("- best parameters:\n")
  cat("gamma =", t$best.parameters$gamma, "; cost =", t$best.parameters$cost, ";\n")
  cat("accuracy:", 100 - t$best.performance * 100, "%\n\n")
  # best.performance は 誤分類率 なので要注意
  plot(t, transform.x=log2, transform.y=log2)
}

############################################################################
setwd(paste0(kBaseDir,"/bin/"))
dataset <- CreateDataset()

# デフォルトパラメータで確認
model <- svm(label ~ ., data = dataset, cross = nrow(dataset))
summary(model)
# Total Accuracy: 64.42953

# おおまかにグリッドサーチ
TuneSVM(dataset, kRoughGammaRange, kRoughCostRange)
# - best parameters:
# gamma = 3.051758e-05 ; cost = 128 ;
# accuracy: 70.33333 %

# 細かなグリッドサーチ
TuneSVM(dataset, kDetailGammaRange, kDetailCostRange)
# - best parameters:
# gamma = 2.157919e-05 ; cost = 64 ;
# accuracy: 70.4698 %

# モデルの生成
model <- svm(
  label ~ .,
  data = dataset,
  gamma = 2^(-15.5),
  cost = 2^(6)
  )

# 新しいデータを分類してみる
gl_samples_df <- CreateSamples("glasses")
result <- predict(model, gl_samples_df)
print(result)
# 1       2       3 
# glasses glasses    none 
# Levels: glasses none

none_samples_df <- CreateSamples("none")
result <- predict(model, none_samples_df)
print(result)
# 1       2       3 
# none    none glasses 
# Levels: glasses none