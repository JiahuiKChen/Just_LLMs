# ============================================================
#  Preference Data Analysis
#  Design: 15 annotators x 4 words x 30 pairs
#  Scale:  0-7 Likert (4 = neutral; >4 = prefer passage 2)
# ============================================================

library(tidyverse)
library(lme4)
library(lmerTest)   # p-values for lmer
library(coin)       # exact Wilcoxon / Friedman tests
library(rstatix)    # tidy Wilcoxon + effect sizes
library(emmeans)    # post-hoc contrasts from LMM


# ── 0. LOAD DATA ─────────────────────────────────────────────
# Expected format: long-form data frame with columns:
#   participant_id : factor, annotator ID
#   word   : factor, one of c("control", "strong", "weak1", "weak2")
#   pair      : integer, 1–30
#   preference_strength    : integer, 0–7

# Adjust this path to your actual file:
df <- read_csv("annotations.csv") %>%
  mutate(across(c(participant_id, word, comparison_id), as.factor))

NEUTRAL <- 4

# ── 1. DESCRIPTIVE STATISTICS ─────────────────────────────────
cat("══════════════════════════════════════════\n")
cat(" 1. DESCRIPTIVE STATISTICS\n")
cat("══════════════════════════════════════════\n")

desc <- df %>%
  group_by(word) %>%
  summarise(
    n      = n(),
    mean   = mean(preference_strength),
    median = median(preference_strength),
    sd     = sd(preference_strength),
    se     = sd / sqrt(n),
    pct_above_neutral = mean(preference_strength > NEUTRAL) * 100,
    .groups = "drop"
  )

print(desc, digits = 3)
cat("\n")


# ── 2. ONE-SAMPLE WILCOXON SIGNED-RANK (vs. neutral = 4) ──────
cat("══════════════════════════════════════════\n")
cat(" 2. ONE-SAMPLE WILCOXON SIGNED-RANK TESTS\n")
cat("    H0: median preference_strength = 4 (neutral)\n")
cat("    H1: median preference_strength > 4 (prefer passage 2)\n")
cat("══════════════════════════════════════════\n")

wilcox_results <- df %>%
  group_by(word) %>%
  summarise(
    wilcox = list(wilcox.test(
      preference_strength,
      mu        = NEUTRAL,
      alternative = "greater",   # one-sided: prefer passage 2
      exact     = FALSE
    )),
    .groups = "drop"
  ) %>%
  mutate(
    W       = map_dbl(wilcox, ~ .x$statistic),
    p.raw   = map_dbl(wilcox, ~ .x$p.value),
    # Effect size r = Z / sqrt(N), Z approximated from W
    N       = map_dbl(wilcox, ~ length(df$preference_strength[df$word == word])),
    Z       = (W - (N * (N + 1) / 4)) / sqrt(N * (N + 1) * (2 * N + 1) / 24),      # one-sided Z
    r       = Z / sqrt(N),
    p.holm  = p.adjust(p.raw, method = "holm")
  ) %>%
  select(word, W, Z, r, p.raw, p.holm)

print(wilcox_results, digits = 4)
cat("\nNote: r < 0.3 small, 0.3–0.5 medium, > 0.5 large\n\n")


# ── 3. FRIEDMAN TEST (differences across words) ────────────
cat("══════════════════════════════════════════\n")
cat(" 3. FRIEDMAN TEST\n")
cat("    Do preference strengths differ across words?\n")
cat("══════════════════════════════════════════\n")

# Friedman needs a subject x condition matrix.
# Here we aggregate per annotator x word (mean over 30 pairs).
df_agg <- df %>%
  group_by(participant_id, word) %>%
  summarise(mean_preference_strength = mean(preference_strength), .groups = "drop")

friedman_mat <- df_agg %>%
  pivot_wider(names_from = word, values_from = mean_preference_strength) %>%
  select(-participant_id) %>%
  as.matrix()

friedman_result <- friedman.test(friedman_mat)
print(friedman_result)

# Effect size: Kendall's W
k <- ncol(friedman_mat)   # number of conditions
n <- nrow(friedman_mat)   # number of subjects
kendall_W <- friedman_result$statistic / (n * (k - 1))
cat(sprintf("Kendall's W = %.3f  (0.1 small, 0.3 medium, 0.5 large)\n\n", kendall_W))


# ── 4. POST-HOC PAIRWISE WILCOXON (with Holm correction) ──────
cat("══════════════════════════════════════════\n")
cat(" 4. POST-HOC PAIRWISE WILCOXON\n")
cat("    (Holm-corrected; run only because Friedman was significant)\n")
cat("══════════════════════════════════════════\n")

# Pairwise on per-annotator means (paired, since same annotators)
posthoc <- df_agg %>%
  pairwise_wilcox_test(
    mean_preference_strength ~ word,
    paired      = TRUE,
    p.adjust.method = "holm",
    alternative = "two.sided"
  )

print(posthoc %>% select(group1, group2, statistic, p, p.adj, p.adj.signif))
cat("\n")


# ── 5. LINEAR MIXED-EFFECTS MODEL ─────────────────────────────
cat("══════════════════════════════════════════\n")
cat(" 5. LINEAR MIXED-EFFECTS MODEL\n")
cat("    Fixed: word; Random: annotator intercept\n")
cat("══════════════════════════════════════════\n")

# Re-level so control is the reference
df <- df %>% mutate(word = relevel(word, ref = "control"))

lmm <- lmer(preference_strength ~ word + (1 | participant_id), data = df, REML = FALSE)
cat("\n--- Fixed Effects ---\n")
print(summary(lmm)$coefficients, digits = 4)

cat("\n--- ANOVA-style test for word effect ---\n")
print(anova(lmm))

# Estimated marginal means (deviation from neutral = 4)
cat("\n--- Estimated Marginal Means (vs. neutral = 4) ---\n")
emm <- emmeans(lmm, ~ word)
print(emm)

# Contrast each word against neutral point (mu = 4)
cat("\n--- One-sample contrasts vs. neutral (4) ---\n")
print(test(emm, null = NEUTRAL, adjust = "holm"))

cat("\n--- Pairwise contrasts (control vs. others) ---\n")
print(contrast(emm, method = "trt.vs.ctrl", ref = "control", adjust = "holm"))


# ── 6. SUMMARY PLOT ───────────────────────────────────────────
cat("\n══════════════════════════════════════════\n")
cat(" 6. GENERATING SUMMARY PLOT\n")
cat("══════════════════════════════════════════\n")

# Significance labels from Wilcoxon (step 2)
sig_labels <- wilcox_results %>%
  mutate(
    label = case_when(
      p.holm < .001 ~ "***",
      p.holm < .01  ~ "**",
      p.holm < .05  ~ "*",
      TRUE          ~ "ns"
    )
  ) %>%
  select(word, label, r)

plot_df <- df %>%
  group_by(word, participant_id) %>%
  summarise(mean_preference_strength = mean(preference_strength), .groups = "drop") %>%
  left_join(sig_labels, by = "word")

p <- ggplot(plot_df, aes(x = word, y = mean_preference_strength, fill = word)) +
  # Boxplot over annotator means
  geom_boxplot(alpha = 0.6, outlier.shape = NA, width = 0.5) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7, color = "grey30") +
  # Neutral reference line
  geom_hline(yintercept = NEUTRAL, linetype = "dashed", color = "firebrick", linewidth = 0.8) +
  annotate("text", x = 0.55, y = NEUTRAL + 0.15, label = "Neutral (4)",
           color = "firebrick", size = 3.5, hjust = 0) +
  # Significance labels
  geom_text(
    data = sig_labels,
    aes(x = word, label = paste0(label, "\nr=", round(r, 2))),
    y = 7.3, inherit.aes = FALSE, size = 3.5, fontface = "bold"
  ) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous(
    limits = c(0.3, 7.7),
    breaks = 1:7,
    labels = c("1\n(strongly\nprefer Removal)", 2:3, "4\n(neutral)", 5:6,
               "7\n(strongly\nprefer Original)")
  ) +
  labs(
    title    = "Passage Preference Ratings by word",
    subtitle = "Boxplots = per-annotator means; dashed line = neutral; labels = Wilcoxon (Holm-corrected)",
    x        = "word",
    y        = "preference_strength"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    panel.grid.minor = element_blank()
  )

ggsave("preference_plot.png", p, width = 8, height = 5.5, dpi = 150)
cat("Plot saved to preference_plot.png\n")