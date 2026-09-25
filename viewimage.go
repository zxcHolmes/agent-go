package agent

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"strings"
)

// ViewImageTool is the name of the built-in tool enabled by Config.ViewImage.
//
// The model calls it with {"url": "https://..."}. The SDK does not download or
// re-encode the image: it answers the tool call with a short text result and
// then adds a user message whose content carries the URL as an "image_url"
// part, so the model sees the image on its next step (tool messages can only
// carry text in OpenAI-compatible APIs). The provider fetches the URL itself,
// so it must be publicly reachable.
const ViewImageTool = "view_image"

func (a *Agent) buildViewImageTool() (json.RawMessage, error) {
	return marshalJSON(map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": ViewImageTool,
			"description": "Look at an image by URL, e.g. one the user sent or one returned by a backend method. " +
				"The image is shown to you in the next user message, right after this tool result.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{"type": "string", "description": "http(s) URL of the image"},
				},
				"required": []string{"url"},
			},
		},
	})
}

// newViewImageCall fills a call for a view_image tool call. It is final right
// away: there is nothing to execute.
func (a *Agent) newViewImageCall(c *RPCCall, arguments string) {
	c.Method, c.Status = ViewImageTool, CallDone
	var args struct {
		URL string `json:"url"`
	}
	err := json.Unmarshal([]byte(arguments), &args)
	if err == nil {
		c.Params, _ = marshalJSON(args)
		err = checkImageURL(args.URL)
	}
	if err != nil {
		c.Result, _ = marshalJSON(map[string]any{"ok": false, "error": "view_image: " + err.Error()})
		return
	}
	c.Result, _ = marshalJSON(map[string]any{"ok": true, "url": args.URL, "note": "The image is attached in the next user message."})
}

func checkImageURL(raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return errors.New(`"url" is required`)
	}
	u, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return errors.New("url must be an absolute http(s) URL")
	}
	return nil
}

// viewImageURL returns the URL of a successful view_image call.
func viewImageURL(c *RPCCall) (string, bool) {
	if c.Method != ViewImageTool || c.Status != CallDone {
		return "", false
	}
	var r struct {
		OK  bool   `json:"ok"`
		URL string `json:"url"`
	}
	if json.Unmarshal(c.Result, &r) != nil || !r.OK {
		return "", false
	}
	return r.URL, true
}

// imageMessageID is deterministic so a crash between writes can be repaired
// without duplicating the message.
func imageMessageID(c *RPCCall) string { return "msg_img_" + c.ID }

// attachImages adds, after all tool messages of the turn, one user message
// per successful view_image call that does not have one yet.
func (a *Agent) attachImages(ctx context.Context, st *runState, calls []RPCCall) error {
	for i := range calls {
		c := &calls[i]
		u, ok := viewImageURL(c)
		if !ok {
			continue
		}
		id := imageMessageID(c)
		if _, err := getMessage(st.db, a.store, a.sessionID, id); err == nil {
			continue
		} else if !errors.Is(err, ErrMessageNotFound) {
			return err
		}
		type imageURL struct {
			URL    string `json:"url"`
			Detail string `json:"detail,omitempty"`
		}
		type part struct {
			Type     string    `json:"type"`
			Text     string    `json:"text,omitempty"`
			ImageURL *imageURL `json:"image_url,omitempty"`
		}
		raw, err := marshalJSON(struct {
			Role    string `json:"role"`
			Content []part `json:"content"`
		}{"user", []part{
			{Type: "text", Text: "[" + ViewImageTool + " result for " + c.ToolCallID + "] " + u},
			{Type: "image_url", ImageURL: &imageURL{URL: u, Detail: a.cfg.ViewImageDetail}},
		}})
		if err != nil {
			return err
		}
		m := newMessage(a.sessionID, raw, MessageDone)
		m.ID, m.Kind, m.ToolCallID = id, MessageKindViewImage, c.ToolCallID
		if err := a.insertMessage(ctx, st, &m); err != nil {
			return err
		}
	}
	return nil
}
