library(ggplot2)
library(patchwork)
library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(svglite)
library(ragg)
library(survival)

root <- "C:/Users/kuku/Desktop/cancer-imaging-ct/paper_v1"
src <- file.path(root, "figures", "source_data")
out <- file.path(root, "figures")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

okabe_ito <- c(LUNG1 = "#0072B2", Radiogenomics = "#E69F00",
               Deep = "#0072B2", Joint = "#D55E00",
               original = "#0072B2", recalibrated = "#D55E00")

theme_paper <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.2646, colour = "black"),
      axis.ticks = element_line(linewidth = 0.2646, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.5),
      legend.key = element_blank(),
      plot.title = element_text(size = base_size + 0.5, face = "bold", hjust = 0.5),
      plot.tag = element_text(size = base_size + 1, face = "bold", hjust = 0, vjust = 1),
      plot.margin = margin(3, 4, 3, 4)
    )
}

save_all <- function(p, stem, width_mm, height_mm, dpi = 600) {
  w <- width_mm / 25.4; h <- height_mm / 25.4
  svglite::svglite(file.path(out, paste0(stem, ".svg")), width = w, height = h)
  print(p); dev.off()
  grDevices::cairo_pdf(file.path(out, paste0(stem, ".pdf")), width = w, height = h, family = "Arial")
  print(p); dev.off()
  ragg::agg_png(file.path(out, paste0(stem, ".png")), width = w, height = h, units = "in", res = dpi)
  print(p); dev.off()
}

# Fig 2: PCA distributions, shared axes within radiomics and embedding pairs.
pca <- read_csv(file.path(src, "fig2_pca_coordinates.csv"), show_col_types = FALSE)
make_pca <- function(panel_id, xlim, ylim) {
  d <- pca %>% filter(panel == panel_id)
  ggplot(d, aes(PC1, PC2, colour = cohort, shape = cohort)) +
    geom_point(size = 1.35, alpha = 0.62, stroke = 0) +
    scale_colour_manual(values = okabe_ito[c("LUNG1", "Radiogenomics")]) +
    scale_shape_manual(values = c(LUNG1 = 16, Radiogenomics = 17)) +
    coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE) +
    labs(title = unique(d$block), x = "PC1", y = "PC2") +
    theme_paper() +
    theme(legend.position = "none", plot.title = element_text(face = "bold", hjust = 0.5))
}
pad_range <- function(x) { r <- range(x); r + c(-0.04, 0.04) * diff(r) }
rad_lim <- list(x = pad_range(pca %>% filter(block == "radiomics") %>% pull(PC1)),
                y = pad_range(pca %>% filter(block == "radiomics") %>% pull(PC2)))
emb_lim <- list(x = pad_range(pca %>% filter(block == "embeddings") %>% pull(PC1)),
                y = pad_range(pca %>% filter(block == "embeddings") %>% pull(PC2)))
p2a <- make_pca("a", rad_lim$x, rad_lim$y) + ggtitle("Radiomics - before ComBat")
p2b <- make_pca("b", rad_lim$x, rad_lim$y) + ggtitle("Radiomics - after ComBat")
p2c <- make_pca("c", emb_lim$x, emb_lim$y) + ggtitle("Embeddings - before ComBat")
p2d <- make_pca("d", emb_lim$x, emb_lim$y) + ggtitle("Embeddings - after ComBat")
fig2 <- (p2a | p2b) / (p2c | p2d) +
  plot_annotation(tag_levels = "a") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(face = "bold"), legend.position = "bottom")
save_all(fig2, "Fig2", 180, 142)

# Fig 3: calibration panels, fixed 0-1 axes and shared legend.
cal <- read_csv(file.path(src, "fig3_calibration_points.csv"), show_col_types = FALSE)
cal <- cal %>% mutate(day = factor(day, levels = c(365, 730, 1095)),
                      row = case_when(grepl("External original", condition) ~ "External original",
                                      grepl("External recalibrated", condition) ~ "External recalibrated",
                                      TRUE ~ "Internal LUNG1 out-of-fold"),
                      state = if_else(grepl("recalibrated", condition), "recalibrated", "original"))
make_cal <- function(row_name, day_name) {
  d <- cal %>% filter(row == row_name, day == day_name)
  ggplot(d, aes(predicted_survival, observed_survival_KM, colour = state, shape = state)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.2646) +
    geom_point(size = 1.8, stroke = 0.25) +
    scale_colour_manual(values = c(original = "#0072B2", recalibrated = "#D55E00")) +
    scale_shape_manual(values = c(original = 16, recalibrated = 1)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = paste0(as.numeric(as.character(day_name))/365, " year"),
         x = "Predicted survival", y = paste0(row_name, "\nObserved survival")) + theme_paper() +
    theme(legend.position = "none")
}
cal_panels <- lapply(c("External original", "External recalibrated", "Internal LUNG1 out-of-fold"),
                     function(r) lapply(c(365, 730, 1095), function(d) make_cal(r, d)))
fig3 <- (cal_panels[[1]][[1]] | cal_panels[[1]][[2]] | cal_panels[[1]][[3]]) /
        (cal_panels[[2]][[1]] | cal_panels[[2]][[2]] | cal_panels[[2]][[3]]) /
        (cal_panels[[3]][[1]] | cal_panels[[3]][[2]] | cal_panels[[3]][[3]]) +
        plot_annotation(tag_levels = "a") & theme(plot.tag = element_text(face = "bold"))
save_all(fig3, "Fig3", 180, 176)

# Fig 4: DCA with rows for evaluation condition and columns for time point.
dca <- read_csv(file.path(src, "fig4_dca_curves.csv"), show_col_types = FALSE)
dca <- dca %>% mutate(day = factor(day, levels = c(365, 730, 1095)),
                      condition = factor(condition, levels = c("Transductive", "Inductive", "Internal OOF")))
make_dca <- function(cond, day_name) {
  d <- dca %>% filter(condition == cond, day == day_name)
  ref <- d %>% distinct(threshold, treat_all, treat_none)
  ggplot(d, aes(threshold, net_benefit, colour = model, fill = model, group = model)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.12, colour = NA) +
    geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.2646) +
    geom_line(linewidth = 0.44) +
    geom_line(data = ref, aes(threshold, treat_all), inherit.aes = FALSE,
              colour = "black", linetype = "dashed", linewidth = 0.35) +
    scale_colour_manual(values = c(Deep = "#0072B2", Joint = "#D55E00")) +
    scale_fill_manual(values = c(Deep = "#0072B2", Joint = "#D55E00")) +
    coord_cartesian(xlim = c(0.05, 0.90), ylim = c(-0.20, 0.60), expand = FALSE) +
    labs(title = paste0(as.numeric(as.character(day_name))/365, " year"),
         x = "Threshold probability", y = paste0(cond, "\nNet benefit")) + theme_paper() +
    theme(legend.position = "bottom")
}
dca_panels <- lapply(levels(dca$condition), function(cn) lapply(levels(dca$day), function(dd) make_dca(cn, dd)))
fig4 <- (dca_panels[[1]][[1]] | dca_panels[[1]][[2]] | dca_panels[[1]][[3]]) /
        (dca_panels[[2]][[1]] | dca_panels[[2]][[2]] | dca_panels[[2]][[3]]) /
        (dca_panels[[3]][[1]] | dca_panels[[3]][[2]] | dca_panels[[3]][[3]]) +
        plot_layout(guides = "collect") + plot_annotation(tag_levels = "a") &
        theme(plot.tag = element_text(face = "bold"), legend.position = "bottom")
save_all(fig4, "Fig4", 180, 176)

# Fig 5: coefficient agreement and paired top components.
coef <- read_csv(file.path(src, "fig5_coefficients.csv"), show_col_types = FALSE)
top_l <- coef %>% slice_max(abs(coef_LUNG1), n = 10, with_ties = FALSE) %>% pull(component)
top_r <- coef %>% slice_max(abs(coef_Radiogenomics), n = 10, with_ties = FALSE) %>% pull(component)
top_union <- union(top_l, top_r)
coef <- coef %>% mutate(group = if_else(component %in% top_union, "Top-10 union", "Other"))
p5a <- ggplot(coef, aes(coef_LUNG1, coef_Radiogenomics, colour = group, shape = group)) +
  geom_hline(yintercept = 0, colour = "grey65", linewidth = 0.2646) +
  geom_vline(xintercept = 0, colour = "grey65", linewidth = 0.2646) +
  geom_abline(linetype = "dashed", linewidth = 0.2646) +
  geom_point(size = 1.7, stroke = 0.2) +
  scale_colour_manual(values = c(`Top-10 union` = "#E69F00", Other = "#9E9E9E")) +
  scale_shape_manual(values = c(`Top-10 union` = 15, Other = 16)) +
  labs(x = "Cox coefficient, LUNG1", y = "Cox coefficient, Radiogenomics") +
  theme_paper() + theme(legend.position = "bottom")
bar_wide <- coef %>% filter(component %in% top_l) %>%
  select(component, coef_LUNG1, coef_Radiogenomics)
bar <- rbind(
  data.frame(component = bar_wide$component, coefficient = bar_wide$coef_LUNG1, cohort = "LUNG1"),
  data.frame(component = bar_wide$component, coefficient = bar_wide$coef_Radiogenomics, cohort = "Radiogenomics")
) %>%
  mutate(component = factor(component, levels = rev(top_l)),
         cohort = factor(cohort, levels = c("LUNG1", "Radiogenomics")))
p5b <- ggplot(bar, aes(coefficient, component, fill = cohort)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62, colour = "black", linewidth = 0.15) +
  scale_fill_manual(values = c(LUNG1 = "#0072B2", Radiogenomics = "#E69F00")) +
  labs(x = "Cox coefficient", y = NULL) + theme_paper() +
  theme(legend.position = "bottom")
fig5 <- (p5a | p5b) + plot_annotation(tag_levels = "a")
save_all(fig5, "Fig5", 180, 92)

# Fig 6: internal-to-external transport forest and paired attenuation.
parse_est <- function(x) as.numeric(stringr::str_extract(x, "^-?[0-9.]+"))
parse_lo <- function(x) as.numeric(stringr::str_match(x, "\\(([0-9.]+)[–-]([0-9.]+)\\)")[,2])
parse_hi <- function(x) as.numeric(stringr::str_match(x, "\\(([0-9.]+)[–-]([0-9.]+)\\)")[,3])
t2 <- read_csv(file.path(root, "tables", "Table2.csv"), show_col_types = FALSE)
names(t2)[2:3] <- c("internal", "external")
forest <- bind_rows(
  t2 %>% transmute(Model, Setting="Internal OOB", estimate=parse_est(internal), lo=parse_lo(internal), hi=parse_hi(internal)),
  t2 %>% transmute(Model, Setting="Target cohort", estimate=parse_est(external), lo=parse_lo(external), hi=parse_hi(external))
) %>% mutate(Model=factor(Model, levels=rev(c("Clinical","Radiomics","Embedding","Joint"))),
             Setting=factor(Setting, levels=c("Internal OOB","Target cohort")))
p6a <- ggplot(forest, aes(estimate, Model, colour=Model, shape=Setting)) +
  geom_vline(xintercept=.5, linetype="dashed", colour="grey55", linewidth=.2646) +
  geom_errorbar(aes(xmin=lo,xmax=hi),orientation="y",width=.16,linewidth=.2646,
                position=position_dodge(width=.42),na.rm=TRUE) +
  geom_point(size=2,stroke=.45,position=position_dodge(width=.42)) +
  scale_colour_manual(values=c(Clinical="#0072B2",Radiomics="#E69F00",Embedding="#009E73",Joint="#D55E00")) +
  scale_shape_manual(values=c(`Internal OOB`=16,`Target cohort`=17)) +
  coord_cartesian(xlim=c(.35,.70)) + labs(x="Harrell's C-index (95% interval)",y=NULL) +
  guides(colour="none") + theme_paper() + theme(legend.position="bottom")
paired <- t2 %>% transmute(Model,Internal=parse_est(internal),`Target cohort`=parse_est(external)) %>%
  tidyr::pivot_longer(c(Internal,`Target cohort`),names_to="Setting",values_to="Cindex") %>%
  mutate(Setting=factor(Setting,levels=c("Internal","Target cohort")),Model=factor(Model,levels=c("Clinical","Radiomics","Embedding","Joint")))
p6b <- ggplot(paired,aes(Setting,Cindex,group=Model,colour=Model)) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey55",linewidth=.2646) +
  geom_line(linewidth=.44) + geom_point(size=1.8) +
  scale_colour_manual(values=c(Clinical="#0072B2",Radiomics="#E69F00",Embedding="#009E73",Joint="#D55E00")) +
  coord_cartesian(ylim=c(.40,.62)) + labs(x=NULL,y="C-index") + theme_paper() + theme(legend.position="bottom")
fig6 <- (p6a | p6b) + plot_layout(widths=c(1.25,1)) +
  plot_annotation(tag_levels="a")
save_all(fig6,"Fig6",180,92)
write_csv(forest,file.path(src,"fig6_transport.csv"))

# Fig 7: stability intervention against size-matched random subsets.
t6 <- read_csv(file.path(root,"tables","Table6.csv"),show_col_types=FALSE)
names(t6) <- c("Threshold","Components","Stable","RandomText","Difference","AbsZ")
t6 <- t6 %>% mutate(RandomMean=as.numeric(stringr::str_extract(RandomText,"^[0-9.]+")),
                    RandomSD=as.numeric(stringr::str_match(RandomText,"±\\s*([0-9.]+)")[,2]),
                    Threshold=factor(sprintf("%.2f",as.numeric(Threshold)),levels=sprintf("%.2f",sort(as.numeric(Threshold)))))
long7 <- bind_rows(
  t6 %>% transmute(Threshold,Components,Method="Stable",Estimate=Stable,Lo=Stable,Hi=Stable),
  t6 %>% transmute(Threshold,Components,Method="Random",Estimate=RandomMean,Lo=RandomMean-RandomSD,Hi=RandomMean+RandomSD)
) %>% mutate(Method=factor(Method,levels=c("Random","Stable")))
p7a <- ggplot(long7,aes(Threshold,Estimate,colour=Method,shape=Method,group=Method)) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey55",linewidth=.2646) +
  geom_errorbar(aes(ymin=Lo,ymax=Hi),width=.10,linewidth=.2646,position=position_dodge(width=.18)) +
  geom_line(linewidth=.44,position=position_dodge(width=.18)) + geom_point(size=1.9,position=position_dodge(width=.18)) +
  scale_colour_manual(values=c(Random="#999999",Stable="#0072B2")) + scale_shape_manual(values=c(Random=1,Stable=16)) +
  labs(x="Stability threshold",y="Target-cohort C-index") + theme_paper() + theme(legend.position="bottom")
p7b <- ggplot(t6,aes(Threshold,as.numeric(Difference),fill=as.numeric(Difference)>0)) +
  geom_hline(yintercept=0,colour="grey40",linewidth=.2646) + geom_col(width=.62,colour="black",linewidth=.15) +
  geom_text(aes(label=paste0("n=",Components)),vjust=ifelse(as.numeric(t6$Difference)>0,-.55,1.35),size=2.25,family="Arial") +
  scale_fill_manual(values=c(`TRUE`="#009E73",`FALSE`="#D55E00"),guide="none") +
  coord_cartesian(ylim=c(-.06,.055)) + labs(x="Stability threshold",y="Stable - random C-index") + theme_paper()
fig7 <- (p7a | p7b) + plot_annotation(tag_levels="a")
save_all(fig7,"Fig7",180,92)
write_csv(long7,file.path(src,"fig7_stability.csv"))

# Supplementary Fig S6: descriptive cohort KM curves, not model-based risk strata.
clin <- read_csv(file.path(root,"pack","master_clinical.csv"),show_col_types=FALSE) %>%
  filter(dataset %in% c("LUNG1","Radiogenomics"),
         !is.na(survival_time_days),!is.na(event)) %>%
  mutate(time_years=survival_time_days/365.25,event=as.integer(event),
         dataset=factor(dataset,levels=c("LUNG1","Radiogenomics")))
sf <- survfit(Surv(time_years,event)~dataset,data=clin)
ss <- summary(sf)
km <- tibble(time=ss$time,surv=ss$surv,lower=ss$lower,upper=ss$upper,
             strata=sub("dataset=","",ss$strata)) %>%
  mutate(strata=recode(strata,LUNG1="LUNG1 (n=422)",
                       Radiogenomics="Radiogenomics (n=211)"))
pal_km <- c(`LUNG1 (n=422)`="#0072B2",`Radiogenomics (n=211)`="#E69F00")
p_s6 <- ggplot(km,aes(time,surv,colour=strata,fill=strata)) +
  geom_ribbon(aes(ymin=lower,ymax=upper),alpha=.12,colour=NA) +
  geom_step(linewidth=.441) +
  coord_cartesian(xlim=c(0,8),ylim=c(0,1),expand=FALSE) +
  scale_colour_manual(values=pal_km) + scale_fill_manual(values=pal_km) +
  labs(x="Time from baseline (years)",y="Overall survival probability") +
  theme_paper() + theme(legend.position="bottom")
save_all(p_s6,"Supplementary_Fig_S6_cohort_KM",180,92)
write_csv(km,file.path(src,"supplementary_fig_s6_cohort_km.csv"))

writeLines(capture.output(sessionInfo()), file.path(out, "R_sessionInfo.txt"))
