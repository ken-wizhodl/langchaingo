package main

import (
	"context"
	"fmt"
	"log"

	em_openai "github.com/tmc/langchaingo/embeddings/openai"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

func main() {
	// Create an embeddings client using the OpenAI API. Requires environment variable OPENAI_API_KEY to be set.
	client, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	e, err := em_openai.NewOpenAI(
		em_openai.WithClient(*client),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Create a new Pinecone vector store.
	store, err := qdrant.New(
		ctx,
		qdrant.WithEmbedder(e),
		qdrant.WithCollectionName("cities"),
	)
	if err != nil {
		log.Fatal(err)
	}

	err = store.AddDocuments(context.Background(), []schema.Document{
		{
			PageContent: "Tokyo",
			Metadata: map[string]any{
				"country":    "Japan",
				"population": 38,
				"area":       2190,
			},
		},
		{
			PageContent: "Paris",
			Metadata: map[string]any{
				"country":    "France",
				"population": 11,
				"area":       105,
			},
		},
		{
			PageContent: "London",
			Metadata: map[string]any{
				"country":    "United Kingdom",
				"population": 9.5,
				"area":       1572,
			},
		},
		{
			PageContent: "Santiago",
			Metadata: map[string]any{
				"country":    "Chile",
				"population": 6.9,
				"area":       641,
			},
		},
		{
			PageContent: "Buenos Aires",
			Metadata: map[string]any{
				"country":    "Argentina",
				"population": 15.5,
				"area":       203,
			},
		},
		{
			PageContent: "Rio de Janeiro",
			Metadata: map[string]any{
				"country":    "Brazil",
				"population": 13.7,
				"area":       1200,
			},
		},
		{
			PageContent: "Sao Paulo",
			Metadata: map[string]any{
				"country":    "Brazil",
				"population": 22.6,
				"area":       1523,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Search for similar documents.
	docs, err := store.SimilaritySearch(ctx, "japan", 1)
	fmt.Println(docs)

	// Search for similar documents using score threshold.
	docs, err = store.SimilaritySearch(ctx, "only cities in south america", 10, vectorstores.WithScoreThreshold(0.80))
	fmt.Println(docs)

	// Search for similar documents using score threshold and metadata filter.
	filter := store.NewMustEqualFilter("country", "Brazil")
	fmt.Println(filter)

	docs, err = store.SimilaritySearch(ctx, "name has Paulo",
		10,
		vectorstores.WithScoreThreshold(0.80),
		vectorstores.WithFilters(filter))
	fmt.Println(docs)
}
