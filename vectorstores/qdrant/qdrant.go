package qdrant

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

var (
	// ErrMissingTextKey is returned in SimilaritySearch if a vector
	// from the query is missing the text key.
	ErrMissingTextKey = errors.New("missing text key in vector metadata")
	// ErrEmbedderWrongNumberVectors is returned when if the embedder returns a number
	// of vectors that is not equal to the number of documents given.
	ErrEmbedderWrongNumberVectors = errors.New(
		"number of vectors from embedder does not match number of documents",
	)
	ErrEmbedderIsNil = errors.New("embedder is NilEmbedder")
	// ErrEmptyResponse is returned if the API gives an empty response.
	ErrEmptyResponse         = errors.New("empty response")
	ErrInvalidScoreThreshold = errors.New(
		"score threshold must be between 0 and 1")
)

// Store is a wrapper around the pinecone rest API and grpc client.
type Store struct {
	embedder embeddings.Embedder

	useCloud         bool
	apiKey           string
	baseURL          string
	collectionName   string
	contentKey       string
	metadataKey      string
	indexKeys        []string
	collectionConfig map[string]any
}

var _ vectorstores.VectorStore = Store{}

// New creates a new Store with options. Options for index name, environment, project name
// and embedder must be set.
func New(ctx context.Context, opts ...Option) (Store, error) {
	s, err := applyClientOptions(opts...)
	if err != nil {
		return Store{}, err
	}

	s.restNewCollection(ctx, s.collectionName)
	for _, indexKey := range s.indexKeys {
		s.restIndexMetadataKey(ctx, s.collectionName, indexKey)
	}

	return s, nil
}

// AddDocuments creates vector embeddings from the documents using the embedder
// and upsert the vectors to the pinecone index.
func (s Store) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) error {
	opts := s.getOptions(options...)
	embedder := s.getEmbedder(opts)

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	vectors, err := embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return err
	}

	// if s.embedder isn't NilEmbedder, then len(vectors) == len(docs)
	_, isNilEmbedder := s.embedder.(vectorstores.NilEmbedder)
	if !isNilEmbedder && len(vectors) != len(docs) {
		return ErrEmbedderWrongNumberVectors
	}

	metadatas := make([]map[string]any, 0, len(docs))
	for i := 0; i < len(docs); i++ {
		metadatas = append(metadatas, docs[i].Metadata)
	}

	return s.restUpsert(ctx, texts, vectors, metadatas, s.collectionName)
}

// SimilaritySearch creates a vector embedding from the query using the embedder
// and queries to find the most similar documents.
func (s Store) SimilaritySearch(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) { //nolint:lll
	opts := s.getOptions(options...)
	embedder := s.getEmbedder(opts)

	filters := s.getFilters(opts)

	scoreThreshold, err := s.getScoreThreshold(opts)
	if err != nil {
		return nil, err
	}

	vector, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	return s.restQuery(ctx, vector, numDocuments, s.collectionName, scoreThreshold,
		filters)
}

// Close closes the grpc connection.
func (s Store) Close() error {
	return nil
}

func (s Store) getScoreThreshold(opts vectorstores.Options) (float32, error) {
	if opts.ScoreThreshold < 0 || opts.ScoreThreshold > 1 {
		return 0, ErrInvalidScoreThreshold
	}
	return opts.ScoreThreshold, nil
}

func (s Store) getFilters(opts vectorstores.Options) any {
	if opts.Filters != nil {
		return opts.Filters
	}

	return nil
}

func (s Store) getOptions(options ...vectorstores.Option) vectorstores.Options {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}
	return opts
}

type FilterMatch struct {
	Key    string
	Values []any
}

func (s Store) NewMustEqualFilter(filters ...FilterMatch) any {
	must := make([]map[string]any, 0, len(filters))
	for _, filter := range filters {
		must = append(must, map[string]any{
			"key": fmt.Sprintf("%s.%s", s.metadataKey, filter.Key),
			"match": map[string]any{
				"any": filter.Values,
			},
		})
	}

	return map[string]any{
		"must": must,
	}
}

func (s Store) getEmbedder(options vectorstores.Options) embeddings.Embedder {
	if options.Embedder != nil {
		return options.Embedder
	}
	return s.embedder
}
