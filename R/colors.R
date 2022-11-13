#' Make colors
#' @description This function creates a color palatte
#' @param n number of colors
#' @param scramble whether to jumble the colors
#' @export
sfc<-function(n, scramble=F){
  if(!n%%1==0)stop("Please input integer n")
  ar20<-c("#16482A", "#1C7C40", "#45AC49", "#69BC9A", "#FBD43F", "#E77A2B", "#DC3F32", "#932528", "#50191E", "#96C4D9", "#2394C4", "#4575AD", "#8681B0", "#6C5492", "#8C4A8D", "#9E2563", "#492C74","#E9E52F", "#F8C566", "#D85191")
  ar10<-ar20[c(1,3,5,7,9,11,13,15,17,20)]
  ar5<-ar10[c(1,3,6,9,10)]
  ar4<-ar10[c(2,4,6,8)]
  if(n>10)funct<-colorRampPalette(ar20)
  if(n<11 & n>4)funct<-colorRampPalette(ar10)
  if(n<5 & n>3)funct<-colorRampPalette(ar4)
  if(n<4)funct<-colorRampPalette(ar5)
  if(scramble){
    return(sample(funct(n),n))
  } else{
    return(funct(n))
  }
}