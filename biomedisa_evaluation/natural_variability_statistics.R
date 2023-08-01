## ---------------------------
##
## Script name: Bee brain statistics
##
## Purpose of script: Supporting Information of Paper
##
## Author: Dr. Coline Monchanin
##
## Email: coline.monchanin@gmail.com
##
## ---------------------------
##
## This script is part of the publication:
##
## Lösel et al. Natural variability in bee brain size and symmetry revealed by micro-CT imaging and deep learning.
##
## Preprint at http://biorxiv.org/lookup/doi/10.1101/2022.10.12.511944 (2022).
##
## ---------------------------

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
HB=read.csv("HBBrain.csv", header=T, sep=";")

#Brain size variation (Table 1)####
(max(HB$Brain,na.rm=T)-min(HB$Brain,na.rm=T))*100/max(HB$Brain,na.rm=T)
(max(HB$AL,na.rm=T)-min(HB$AL,na.rm=T))*100/max(HB$AL,na.rm=T)
(max(HB$MB,na.rm=T)-min(HB$MB,na.rm=T))*100/max(HB$MB,na.rm=T)
(max(HB$OL,na.rm=T)-min(HB$OL,na.rm=T))*100/max(HB$OL,na.rm=T)
(max(HB$ME,na.rm=T)-min(HB$ME,na.rm=T))*100/max(HB$ME,na.rm=T)
(max(HB$LO,na.rm=T)-min(HB$LO,na.rm=T))*100/max(HB$LO,na.rm=T)
(max(HB$CX,na.rm=T)-min(HB$CX,na.rm=T))*100/max(HB$CX,na.rm=T)
(max(HB$OTH,na.rm=T)-min(HB$OTH,na.rm=T))*100/max(HB$OTH,na.rm=T)

#Covariance (Fig 5, S2 Fig, S2 Table)####
rcorr(as.matrix(HB[6:13]))
rcorr(as.matrix(HB[13:20]))

#Between colonies (Table 1 and Fig 6)#####
mod=lmer(Brain~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(AL~Hive+(1|Site),data=HB)
anova(mod)

mod=lmer(MB~Hive+(1|Site),data=HB)
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

#Variability within hives (S3 Table)#####
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
BB=read.csv("BBBrain.csv", header=T, sep=";")

#Brain size variation (S4 Table)#####
(max(BB$Brain,na.rm=T)-min(BB$Brain,na.rm=T))*100/max(BB$Brain,na.rm=T)
(max(BB$AL,na.rm=T)-min(BB$AL,na.rm=T))*100/max(BB$AL,na.rm=T)
(max(BB$MB,na.rm=T)-min(BB$MB,na.rm=T))*100/max(BB$MB,na.rm=T)
(max(BB$OL,na.rm=T)-min(BB$OL,na.rm=T))*100/max(BB$OL,na.rm=T)
(max(BB$ME,na.rm=T)-min(BB$ME,na.rm=T))*100/max(BB$ME,na.rm=T)
(max(BB$LO,na.rm=T)-min(BB$LO,na.rm=T))*100/max(BB$LO,na.rm=T)
(max(BB$CX,na.rm=T)-min(BB$CX,na.rm=T))*100/max(BB$CX,na.rm=T)
(max(BB$OTH,na.rm=T)-min(BB$OTH,na.rm=T))*100/max(BB$OTH,na.rm=T)

########Covariance (S10 Fig, S11 Fig, S5 Table)####
rcorr(as.matrix(BB[4:11]))
rcorr(as.matrix(BB[11:18]))

####Between colonies (S4 Table)#####
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
H1=subset(BB,Hive=="H1")
(max(H1$Brain,na.rm=T)-min(H1$Brain,na.rm=T))*100/max(H1$Brain,na.rm=T)
(max(H1$AL,na.rm=T)-min(H1$AL,na.rm=T))*100/max(H1$AL,na.rm=T)
(max(H1$MB,na.rm=T)-min(H1$MB,na.rm=T))*100/max(H1$MB,na.rm=T)
(max(H1$OL,na.rm=T)-min(H1$OL,na.rm=T))*100/max(H1$OL,na.rm=T)
(max(H1$ME,na.rm=T)-min(H1$ME,na.rm=T))*100/max(H1$ME,na.rm=T)
(max(H1$LO,na.rm=T)-min(H1$LO,na.rm=T))*100/max(H1$LO,na.rm=T)
(max(H1$CX,na.rm=T)-min(H1$CX,na.rm=T))*100/max(H1$CX,na.rm=T)
(max(H1$OTH,na.rm=T)-min(H1$OTH,na.rm=T))*100/max(H1$OTH,na.rm=T)

