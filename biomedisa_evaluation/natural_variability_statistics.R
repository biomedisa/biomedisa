#library(multcomp)
#library(emmeans)
#library(ggplot2)
library(lme4)
#library(ggpubr)
#library(RVAideMemoire)
#library(esquisse)
library(lmerTest)
#library(afex)
#library(car)
#library(glmmTMB)
#library(plyr)
#library(dplyr)
library(Hmisc)
options( warn = -1 )

######ANALYSIS FOR HONEYBEES############

#Brain size variation####
HB=read.csv("HBBrain.csv", header=T, sep=";")
(max(HB$Brain,na.rm=T)-min(HB$Brain,na.rm=T))*100/max(HB$Brain,na.rm=T)



#Covariance####
rcorr(as.matrix(HB[6:13]))
rcorr(as.matrix(HB[13:20]))

#Between colonies#####
mod=lmer(Brain~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(AL~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(MB~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(CX~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(LO~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(ME~Hive+(1|Date),data=HB)
anova(mod)

mod=lmer(OL~Hive+(1|Date),data=HB)
anova(mod)


#Asymmetry####
HB=read.csv("HBAsym.csv", header=T, sep=";")

mod=lmer(AL~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(MB~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(ME~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(LO~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(OL~Side+(1|Date)+(1|ID),data=HB)
anova(mod)


#Asymmetry relative volume####
HB=read.csv("HBAsym.csv", header=T, sep=";")

mod=lmer(ALr~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(MBr~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(MEr~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(LOr~Side+(1|Date)+(1|ID),data=HB)
anova(mod)



#TABLE 1#####
mod=lmer(Brain~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(AL~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(MB~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(MBr~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(OL~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(LO~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(ME~Hive+(1|Site),data=HB)
anova(mod)

#mod=lmer(CX~Hive+(1|Site),data=HB)
#anova(mod)

#mod=lmer(OTH~Hive+(1|Site),data=HB)
#anova(mod)



#TABLE S3: % variability within hives#####
HB=read.csv("HBBrain.csv", header=T, sep=";")

H1=subset(HB,Hive=="H1")
(max(H1$Brain,na.rm=T)-min(H1$Brain,na.rm=T))*100/max(H1$Brain,na.rm=T)
(max(H1$AL,na.rm=T)-min(H1$AL,na.rm=T))*100/max(H1$AL,na.rm=T)
(max(H1$MB,na.rm=T)-min(H1$MB,na.rm=T))*100/max(H1$MB,na.rm=T)
(max(H1$OL,na.rm=T)-min(H1$OL,na.rm=T))*100/max(H1$OL,na.rm=T)
(max(H1$CX,na.rm=T)-min(H1$CX,na.rm=T))*100/max(H1$CX,na.rm=T)
(max(H1$OTH,na.rm=T)-min(H1$OTH,na.rm=T))*100/max(H1$OTH,na.rm=T)
(max(H1$ME,na.rm=T)-min(H1$ME,na.rm=T))*100/max(H1$ME,na.rm=T)
(max(H1$LO,na.rm=T)-min(H1$LO,na.rm=T))*100/max(H1$LO,na.rm=T)



########ANALYSIS FOR BUMBLEBEES##########

#Brain size variation#####
BB=read.csv("BBBrain.csv", header=T, sep=";")

(max(BB$Brain,na.rm=T)-min(BB$Brain,na.rm=T))*100/max(BB$Brain,na.rm=T)
(max(BB$AL,na.rm=T)-min(BB$AL,na.rm=T))*100/max(BB$AL,na.rm=T)
(max(BB$MB,na.rm=T)-min(BB$MB,na.rm=T))*100/max(BB$MB,na.rm=T)
(max(BB$OL,na.rm=T)-min(BB$OL,na.rm=T))*100/max(BB$OL,na.rm=T)
(max(BB$ME,na.rm=T)-min(BB$ME,na.rm=T))*100/max(BB$ME,na.rm=T)
(max(BB$LO,na.rm=T)-min(BB$LO,na.rm=T))*100/max(BB$LO,na.rm=T)
(max(BB$CX,na.rm=T)-min(BB$CX,na.rm=T))*100/max(BB$CX,na.rm=T)
(max(BB$OTH,na.rm=T)-min(BB$OTH,na.rm=T))*100/max(BB$OTH,na.rm=T)



########Covariance####

rcorr(as.matrix(BB[4:11]))
rcorr(as.matrix(BB[11:18]))

####Between colonies#####

mod=lm(Brain~Hive,data=BB)
anova(mod)

mod=lm(AL~Hive,data=BB)
anova(mod)

mod=lm(MB~Hive,data=BB)
anova(mod)

mod=lm(OL~Hive,data=BB)
anova(mod)

mod=lm(ME~Hive,data=BB)
anova(mod)

mod=lm(LO~Hive,data=BB)
anova(mod)

mod=lm(CX~Hive,data=BB)
anova(mod)

mod=lm(OTH~Hive,data=BB)
anova(mod)


####Within colonies######
BB=read.csv("BBBrain.csv", header=T, sep=";")

H1=subset(BB,Hive=="H1")

(max(H1$Brain,na.rm=T)-min(H1$Brain,na.rm=T))*100/max(H1$Brain,na.rm=T)
(max(H1$AL,na.rm=T)-min(H1$AL,na.rm=T))*100/max(H1$AL,na.rm=T)
(max(H1$MB,na.rm=T)-min(H1$MB,na.rm=T))*100/max(H1$MB,na.rm=T)
(max(H1$OL,na.rm=T)-min(H1$OL,na.rm=T))*100/max(H1$OL,na.rm=T)
(max(H1$ME,na.rm=T)-min(H1$ME,na.rm=T))*100/max(H1$ME,na.rm=T)
(max(H1$LO,na.rm=T)-min(H1$LO,na.rm=T))*100/max(H1$LO,na.rm=T)
(max(H1$CX,na.rm=T)-min(H1$CX,na.rm=T))*100/max(H1$CX,na.rm=T)
(max(H1$OTH,na.rm=T)-min(H1$OTH,na.rm=T))*100/max(H1$OTH,na.rm=T)



#Asymmetry#####
BB=read.csv("BBAsym.csv", header=T, sep=";")

mod=lmer(AL~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(MB~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(ME~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(LO~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(OL~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

#Asymmetry relative volume####
BB=read.csv("BBAsym.csv", header=T, sep=";")

mod=lmer(ALr~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(MBr~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(MEr~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(LOr~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

mod=lmer(OLr~Side+(1|ID)+(1|Hive),data=BB)
anova(mod)

