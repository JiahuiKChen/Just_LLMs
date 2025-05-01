from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")
sentences = ['My name is SONAR.', 'I can embed the sentences into vectorial space.']
embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")
print(embeddings.shape)