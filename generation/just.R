library(tidyverse)
library(ggstance)

just_results <- read_tsv("Downloads/all_models_all_particles_generation_candidates.tsv")

model_meta <- tribble(
  ~source_model,~short,~family,~instruction,
  "Meta-Llama-3-8B","Llama3-8B", "Llama3",FALSE,
  "Meta-Llama-3-8B-Instruct","Llama3-8B-I","Llama3",TRUE,
  "OLMo-2-1124-7B", "OLMo2-7B","OLMo2",FALSE,
  "OLMo-2-1124-7B-Instruct", "OLMo2-7B-I","OLMo2",TRUE,
  "gemma-2-9b", "Gemma2-9B", "Gemma2",FALSE,
  "gemma-2-9b-it", "Gemma2-9B-I", "Gemma2",TRUE,
  "Qwen3.5-9B","Qwen3.5-9B","Qwen3.5",FALSE,
  "Qwen3.5-9B-Instruct","Qwen3.5-9B-I","Qwen3.5",TRUE
)

relevant <- just_results %>%
  mutate(
    source_model = case_when(
      source_model == "Qwen3.5-9B" & source_results_dir == "qwen35_instruct" ~ "Qwen3.5-9B-Instruct",
      TRUE ~ source_model
    ),
    word = factor(word, levels = c("not", "just", "only"))
  ) %>%
  select(source_model, word, generated_from, response, own_log_prob_per_token, other_log_prob_per_token, own_context_log_prob_advantage_per_token)

agg_results <- relevant %>%
  group_by(source_model, word, generated_from) %>%
  summarize(
    n = n(),
    std = sd(own_context_log_prob_advantage_per_token),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * std/sqrt(n),
    advantage = mean(own_context_log_prob_advantage_per_token),
    advantage_pct = mean(own_context_log_prob_advantage_per_token > 0)
  ) %>%
  ungroup() %>%
  inner_join(model_meta)

agg_results %>%
  ggplot(aes(advantage, short, color = word, shape = generated_from)) +
  geom_point() +
  geom_linerangeh(aes(xmin=advantage-cb, xmax=advantage+cb))

agg_results %>%
  ggplot(aes(word, advantage, shape = instruction, color = generated_from)) +
  geom_point() +
  geom_linerange(aes(ymin=advantage-cb, ymax=advantage+cb)) +
  facet_wrap(~family)

agg_results %>%
  ggplot(aes(word, advantage_pct, shape = instruction, color = generated_from)) +
  geom_point() +
  facet_wrap(~family)


relevant %>%
  # group_by(source_model, word, generated_from)
  group_by(source_model, word, generated_from) %>%
  nest() %>%
  mutate(
    diff = map(data, function(x){
      test = t.test(x$own_context_log_prob_advantage_per_token) %>%
        broom::tidy()
      
      d = lsr::cohensD(x$own_context_log_prob_advantage_per_token)
      test %>% mutate(effect_size = d)
    })
  ) %>%
  select(-data) %>%
  unnest(diff) %>% 
  ungroup() %>%
  inner_join(model_meta) %>%
  ggplot(aes(word, effect_size, shape = instruction, color = generated_from)) +
  geom_point() +
  facet_wrap(~family)
  
  
  
  