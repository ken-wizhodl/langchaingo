package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/schema"
)

// APIError is an error type returned if the status code from the rest
// api is not 200.
type APIError struct {
	Task    string
	Message string
}

func newAPIError(task string, body io.ReadCloser) APIError {
	buf := new(bytes.Buffer)
	_, err := io.Copy(buf, body)
	if err != nil {
		return APIError{Task: "reading body of error message", Message: err.Error()}
	}

	return APIError{Task: task, Message: buf.String()}
}

func (e APIError) Error() string {
	return fmt.Sprintf("%s: %s", e.Task, e.Message)
}

type point struct {
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload"`
	ID      string         `json:"id"`
}

type upsertPayload struct {
	Points []point `json:"points"`
}

func (s Store) restUpsert(
	ctx context.Context,
	vectors [][]float32,
	metadatas []map[string]any,
	collection string,
) error {
	v := make([]point, 0, len(vectors))
	for i := 0; i < len(vectors); i++ {
		v = append(v, point{
			Vector:  vectors[i],
			Payload: metadatas[i],
			ID:      uuid.New().String(),
		})
	}

	payload := upsertPayload{
		Points: v,
	}

	endpoint := getEndpoint(s.baseURL, collection, "/points")
	body, status, err := doRequest(
		ctx,
		payload,
		endpoint,
		s.apiKey,
		http.MethodPut,
	)
	if err != nil {
		return err
	}
	defer body.Close()

	if status == http.StatusOK {
		return nil
	}

	return newAPIError("upserting vectors", body)
}

type scoredPoint struct {
	ID      string         `json:"id"`
	Version int            `json:"version"`
	Score   float32        `json:"score"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector"`
}

type queriesResponse struct {
	Time   float32       `json:"time"`
	Status string        `json:"status"`
	Result []scoredPoint `json:"result"`
}

type queryPayload struct {
	WithVector     bool      `json:"with_vector"`
	WithPayload    bool      `json:"with_payload"`
	Vector         []float32 `json:"vector"`
	Limit          int       `json:"limit"`
	Filter         any       `json:"filter"`
	ScoreThreshold float32   `json:"score_threshold"`
}

func (s Store) restQuery(
	ctx context.Context,
	vector []float32,
	numVectors int,
	collection string,
	scoreThreshold float32,
	filter any,
) ([]schema.Document, error) {
	payload := queryPayload{
		WithVector:     true,
		WithPayload:    true,
		Vector:         vector,
		Limit:          numVectors,
		Filter:         filter,
		ScoreThreshold: scoreThreshold,
	}

	endpoint := getEndpoint(s.baseURL, collection, "/points/search")
	body, statusCode, err := doRequest(
		ctx,
		payload,
		endpoint,
		s.apiKey,
		http.MethodPost,
	)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	if statusCode != http.StatusOK {
		return nil, newAPIError("querying index", body)
	}

	var response queriesResponse

	decoder := json.NewDecoder(body)
	err = decoder.Decode(&response)
	if err != nil {
		return nil, err
	}

	if len(response.Result) == 0 {
		return nil, ErrEmptyResponse
	}

	docs := make([]schema.Document, 0, len(response.Result))
	for _, spoint := range response.Result {
		pageContent, ok := spoint.Payload[s.textKey].(string)
		if !ok {
			return nil, ErrMissingTextKey
		}
		delete(spoint.Payload, s.textKey)

		doc := schema.Document{
			PageContent: pageContent,
			Metadata:    spoint.Payload,
			Score:       spoint.Score,
		}

		// If scoreThreshold is not 0, we only return matches with a score above the threshold.
		if scoreThreshold != 0 && spoint.Score >= scoreThreshold {
			docs = append(docs, doc)
		} else if scoreThreshold == 0 { // If scoreThreshold is 0, we return all matches.
			docs = append(docs, doc)
		}
	}

	return docs, nil
}

func doRequest(ctx context.Context, payload any, url, apiKey, method string) (io.ReadCloser, int, error) {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, err
	}
	body := bytes.NewReader(payloadBytes)

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, 0, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("accept", "text/plain")
	req.Header.Set("Api-Key", apiKey)

	r, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	return r.Body, r.StatusCode, err
}

func getEndpoint(baseURL, collection, path string) string {
	path = strings.TrimPrefix(path, "/")
	return fmt.Sprintf(
		"%s/collections/%s/%s",
		baseURL,
		collection,
		path,
	)
}
