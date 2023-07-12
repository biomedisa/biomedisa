## ---------------------------
##
## Script name: Bee brain statistics
##
## Purpose of script: Supporting Information of Paper
##
## Author: Dr. Coline Monchanin
##
## Date Created: 2023-06-15
##
## Email: coline.monchanin@gmail.com
##
## ---------------------------
##
## This script is part of the publication:
##
## Philipp D. Lösel, Coline Monchanin, Renaud Lebrun, Alejandra Jayme, Jacob Relle, Jean-Marc Devaud, Vincent Heuveline, Mathieu Lihoreau
## Natural variability in bee brain size and symmetry revealed by micro-CT imaging and deep learning.
## Preprint at http://biorxiv.org/lookup/doi/10.1101/2022.10.12.511944 (2022).
##
## ---------------------------

library(multcomp)
library(emmeans)
library(ggplot2)
library(lme4)
library(ggpubr)
library(RVAideMemoire)
library(esquisse)
library(ggpubr)
library(lmerTest)
library(afex)
library(car)
library(glmmTMB)
library(plyr)
library(dplyr)
library(Hmisc)


######ANALYSIS FOR HONEYBEES############

#Brain size variation####
HB=read.csv("HBBrain.csv", header=T, sep=";")
(max(HB$Brain,na.rm=T)-min(HB$Brain,na.rm=T))*100/max(HB$Brain,na.rm=T)



#Covariance####
rcorr(as.matrix(HB[8:15]))
rcorr(as.matrix(HB[15:22]))

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

mod=lmer(rAL~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod1=lmer(rMB~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(rME~Side+(1|Date)+(1|ID),data=HB)
anova(mod)

mod=lmer(rLO~Side+(1|Date)+(1|ID),data=HB)
anova(mod)



#TABLE 1#####
mod=lmer(Brain~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(AL~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(MB~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(MBR~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(OL~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(LO~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(ME~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(CX~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(OTH~Hive+(1|Site),data=HB)
anova(mod)



#TABLE S3: % variability within hives#####

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

mod=lm(Brain~Colony,data=BB)
anova(mod)

mod=lm(AL~Colony,data=BB)
anova(mod)

mod=lm(MB~Colony,data=BB)
anova(mod)

mod=lm(OL~Colony,data=BB)
anova(mod)

mod=lm(ME~Colony,data=BB)
anova(mod)

mod=lm(LO~Colony,data=BB)
anova(mod)

mod=lm(CX~Colony,data=BB)
anova(mod)

mod=lm(OTH~Colony,data=BB)
anova(mod)


####Within colonies######
HiveA=subset(BB,Colony=="A")

(max(HiveA$Brain,na.rm=T)-min(HiveA$Brain,na.rm=T))*100/max(HiveA$Brain,na.rm=T)
(max(HiveA$AL,na.rm=T)-min(HiveA$AL,na.rm=T))*100/max(HiveA$AL,na.rm=T)
(max(HiveA$MB,na.rm=T)-min(HiveA$MB,na.rm=T))*100/max(HiveA$MB,na.rm=T)
(max(HiveA$OL,na.rm=T)-min(HiveA$OL,na.rm=T))*100/max(HiveA$OL,na.rm=T)
(max(HiveA$ME,na.rm=T)-min(HiveA$ME,na.rm=T))*100/max(HiveA$ME,na.rm=T)
(max(HiveA$LO,na.rm=T)-min(HiveA$LO,na.rm=T))*100/max(HiveA$LO,na.rm=T)
(max(HiveA$CX,na.rm=T)-min(HiveA$CX,na.rm=T))*100/max(HiveA$CX,na.rm=T)
(max(HiveA$OTH,na.rm=T)-min(HiveA$OTH,na.rm=T))*100/max(HiveA$OTH,na.rm=T)



#Asymmetry#####

mod1=lmer(AL~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(MB~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(ME~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(LO~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(OL~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

#Asymmetry relative volume####

mod1=lmer(rAL~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(rMB~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(rME~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(rLO~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

mod1=lmer(rOL~Side+(1|ID)+(1|Colony),data=BB)
anova(mod1)

