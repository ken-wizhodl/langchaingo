package qdrant

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/embeddings"
)

const (
	_qdrantKeyEnvVarName = "QDRANT_API_KEY"
	_qdrantURLEnvVarName = "QDRANT_BASE_URL"
	_defaultTextKey      = "text"
)

// ErrInvalidOptions is returned when the options given are invalid.
var ErrInvalidOptions = errors.New("invalid options")

// Option is a function type that can be used to modify the client.
type Option func(p *Store)

// WithEmbedder is an option for setting the embedder to use. Must be set.
func WithEmbedder(e embeddings.Embedder) Option {
	return func(p *Store) {
		p.embedder = e
	}
}

// WithAPIKey is an option for setting the api key. If the option is not set
// the api key is read from the PINECONE_API_KEY environment variable. If the
// variable is not present, an error will be returned.
func WithAPIKey(apiKey string) Option {
	return func(p *Store) {
		p.apiKey = apiKey
	}
}

// WithBaseURL is an option for setting the base url. If the option is not set
// the base url is read from the QDRANT_BASE_URL environment variable. If the
// variable is not present, an error will be returned.
func WithBaseURL(baseURL string) Option {
	return func(p *Store) {
		p.baseURL = baseURL
	}
}

// WithUseCloud is an option for setting if it's the qdrant cloud or not.
func WithUseCloud(isCloud bool) Option {
	return func(p *Store) {
		p.useCloud = isCloud
	}
}

// WithTextKey is an option for setting the text key in the metadata to the vectors
// in the index. The text key stores the text of the document the vector represents.
func WithTextKey(textKey string) Option {
	return func(p *Store) {
		p.textKey = textKey
	}
}

// NameSpace is an option for setting the nameSpace to upsert and query the vectors
// from. Must be set.
func WithCollectionName(collectionName string) Option {
	return func(p *Store) {
		p.collectionName = collectionName
	}
}

func applyClientOptions(opts ...Option) (Store, error) {
	o := &Store{
		textKey: _defaultTextKey,
	}

	for _, opt := range opts {
		opt(o)
	}

	if o.embedder == nil {
		return Store{}, fmt.Errorf("%w: missing embedder", ErrInvalidOptions)
	}

	if o.apiKey == "" {
		o.apiKey = os.Getenv(_qdrantKeyEnvVarName)
		if o.useCloud && o.apiKey == "" {
			return Store{}, fmt.Errorf(
				"%w: missing api key. Pass it as an option or set the %s environment variable",
				ErrInvalidOptions,
				_qdrantKeyEnvVarName,
			)
		}
	}

	if o.baseURL == "" {
		o.baseURL = os.Getenv(_qdrantURLEnvVarName)
		if o.baseURL == "" {
			return Store{}, fmt.Errorf(
				"%w: missing api url. Pass it as an option or set the %s environment variable",
				ErrInvalidOptions,
				_qdrantURLEnvVarName,
			)
		}
	}

	if o.collectionName == "" {
		return Store{}, fmt.Errorf("%w: missing collection name", ErrInvalidOptions)
	}

	return *o, nil
}
