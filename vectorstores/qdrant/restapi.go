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
	texts []string,
	vectors [][]float32,
	metadatas []map[string]any,
	collection string,
) error {
	v := make([]point, 0, len(vectors))
	for i := 0; i < len(vectors); i++ {
		ID := uuid.New().String()
		if metadatas[i] != nil && metadatas[i]["__point_id"] != nil {
			ID = metadatas[i]["__point_id"].(string)
		}
		v = append(v, point{
			Vector:  vectors[i],
			Payload: map[string]any{s.contentKey: texts[i], s.metadataKey: metadatas[i]},
			ID:      ID,
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

func (s Store) restDeletePoints(ctx context.Context, collection string, filter any) error {
	payload := map[string]any{
		"filter": filter,
	}
	endpoint := getEndpoint(s.baseURL, collection, "/points/delete")
	body, statusCode, err := doRequest(
		ctx,
		payload,
		endpoint,
		s.apiKey,
		http.MethodPost,
	)
	if err != nil {
		return err
	}
	defer body.Close()

	if statusCode == http.StatusOK {
		return nil
	} else {
		return newAPIError("deleting points", body)
	}
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

func (s Store) restNewCollection(ctx context.Context, collection string) error {
	endpoint := getEndpoint(s.baseURL, collection, "")
	config := map[string]any{}
	for k, v := range s.collectionConfig {
		config[k] = v
	}
	config["name"] = collection
	body, status, err := doRequest(
		ctx,
		config,
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

	return newAPIError("creating collection", body)
}

func (s Store) restIndexMetadataKey(ctx context.Context, collection, key string) error {
	endpoint := getEndpoint(s.baseURL, collection, "/index")
	body, status, err := doRequest(
		ctx,
		map[string]string{
			"field_name":   key,
			"field_schema": "keyword",
		},
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

	return newAPIError("indexing metadata key", body)
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
		pageContent, ok := spoint.Payload[s.contentKey].(string)
		if !ok {
			return nil, ErrMissingTextKey
		}

		doc := schema.Document{
			PageContent: pageContent,
			Metadata:    spoint.Payload[s.metadataKey].(map[string]any),
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

type scrollResult struct {
	Points         []point `json:"points"`
	NextPageOffset *string `json:"next_page_offset"`
}

type scrollResponse struct {
	Time   float32       `json:"time"`
	Status string        `json:"status"`
	Result *scrollResult `json:"result"`
}

type ScrollPointsRequest struct {
	Offset      string `json:"offset,omitempty"`
	Limit       int    `json:"limit"`
	Filter      any    `json:"filter"`
	WithPayload bool   `json:"with_payload"`
	WithVector  bool   `json:"with_vector"`
}

func (s Store) restScrollPoints(ctx context.Context, collection string, req *ScrollPointsRequest) ([]schema.Document, string, error) {
	endpoint := getEndpoint(s.baseURL, collection, "/points/scroll")
	body, statusCode, err := doRequest(
		ctx,
		req,
		endpoint,
		s.apiKey,
		http.MethodPost,
	)
	if err != nil {
		return nil, "", err
	}
	defer body.Close()

	if statusCode != http.StatusOK {
		return nil, "", newAPIError("scrolling points", body)
	}

	var response scrollResponse

	decoder := json.NewDecoder(body)
	err = decoder.Decode(&response)
	if err != nil {
		return nil, "", err
	}

	points := response.Result.Points

	docs := make([]schema.Document, 0, len(points))
	for _, spoint := range points {
		pageContent, ok := spoint.Payload[s.contentKey].(string)
		if !ok {
			return nil, "", ErrMissingTextKey
		}

		doc := schema.Document{
			PageContent: pageContent,
			Metadata:    spoint.Payload[s.metadataKey].(map[string]any),
		}

		docs = append(docs, doc)
	}

	nextOffset := ""
	if response.Result.NextPageOffset != nil {
		nextOffset = *response.Result.NextPageOffset
	}

	return docs, nextOffset, nil
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
	if path != "" && !strings.HasPrefix(path, "/") {
		path = fmt.Sprintf("/%s", path)
	}
	return fmt.Sprintf(
		"%s/collections/%s%s",
		baseURL,
		collection,
		path,
	)
}
