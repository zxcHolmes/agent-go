package agent

import (
	"bytes"
	"encoding"
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// ParamsSchema generates a JSON Schema for P from its fields and struct tags,
// in field declaration order:
//
//	type RefundParams struct {
//		OrderID string   `json:"order_id" desc:"Order id, e.g. ORD-123"`
//		Mode    string   `json:"mode" enum:"full,partial" desc:"Refund mode"`
//		Amount  float64  `json:"amount,omitempty" desc:"Amount for partial refunds"`
//		Items   []string `json:"items" required:"false"`
//	}
//
// Tags:
//   - json: field name; "-" skips the field. Fields follow encoding/json rules
//     (unexported fields are skipped, embedded structs are flattened).
//   - desc: description shown to the model.
//   - enum: comma-separated allowed values.
//   - required: "true" or "false". By default a field is required unless it
//     is a pointer or its json tag has omitempty / omitzero.
//
// Supported types: strings, bools, integers, floats, slices, arrays, maps with
// string keys, nested structs, pointers, time.Time (date-time string),
// encoding.TextMarshaler (string), and any / json.RawMessage (any JSON value).
func ParamsSchema[P any]() json.RawMessage {
	t := reflect.TypeOf((*P)(nil)).Elem()
	b, _ := marshalJSON(schemaFor(t, map[reflect.Type]bool{}))
	return b
}

type jsonSchema struct {
	Type                 string       `json:"type,omitempty"`
	Format               string       `json:"format,omitempty"`
	Description          string       `json:"description,omitempty"`
	Enum                 []any        `json:"enum,omitempty"`
	Items                *jsonSchema  `json:"items,omitempty"`
	Properties           *schemaProps `json:"properties,omitempty"`
	Required             []string     `json:"required,omitempty"`
	AdditionalProperties *jsonSchema  `json:"additionalProperties,omitempty"`
}

type schemaProp struct {
	name   string
	schema *jsonSchema
}

// schemaProps keeps declaration order, which a Go map would lose.
type schemaProps []schemaProp

func (p schemaProps) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, prop := range p {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := marshalJSON(prop.name)
		if err != nil {
			return nil, err
		}
		v, err := marshalJSON(prop.schema)
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

var (
	timeType          = reflect.TypeOf(time.Time{})
	rawMessageType    = reflect.TypeOf(json.RawMessage(nil))
	textMarshalerType = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
)

func schemaFor(t reflect.Type, seen map[reflect.Type]bool) *jsonSchema {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch {
	case t == timeType:
		return &jsonSchema{Type: "string", Format: "date-time"}
	case t == rawMessageType:
		return &jsonSchema{}
	case t.Implements(textMarshalerType) || reflect.PointerTo(t).Implements(textMarshalerType):
		return &jsonSchema{Type: "string"}
	}
	switch t.Kind() {
	case reflect.String:
		return &jsonSchema{Type: "string"}
	case reflect.Bool:
		return &jsonSchema{Type: "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &jsonSchema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return &jsonSchema{Type: "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			return &jsonSchema{Type: "string", Description: "base64"}
		}
		return &jsonSchema{Type: "array", Items: schemaFor(t.Elem(), seen)}
	case reflect.Map:
		return &jsonSchema{Type: "object", AdditionalProperties: schemaFor(t.Elem(), seen)}
	case reflect.Struct:
		if seen[t] { // recursive type
			return &jsonSchema{Type: "object"}
		}
		seen[t] = true
		defer delete(seen, t)
		s := &jsonSchema{Type: "object"}
		props := schemaProps{}
		addStructFields(t, s, &props, seen)
		if len(props) > 0 {
			s.Properties = &props
		}
		return s
	default: // interface and anything else: any JSON value
		return &jsonSchema{}
	}
}

func addStructFields(t reflect.Type, s *jsonSchema, props *schemaProps, seen map[reflect.Type]bool) {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		ft := f.Type
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		if f.Anonymous && name == "" && ft.Kind() == reflect.Struct {
			addStructFields(ft, s, props, seen) // flattened like encoding/json
			continue
		}
		if !f.IsExported() {
			continue
		}
		if name == "" {
			name = f.Name
		}
		fs := schemaFor(f.Type, seen)
		if hasOpt(opts, "string") && fs.Type != "" && fs.Type != "string" {
			fs = &jsonSchema{Type: "string"}
		}
		fs.Description = strings.TrimSpace(strings.TrimSpace(fs.Description + " " + f.Tag.Get("desc")))
		if enum := f.Tag.Get("enum"); enum != "" {
			fs.Enum = parseEnum(enum, fs.Type)
		}
		required := f.Type.Kind() != reflect.Pointer && !hasOpt(opts, "omitempty") && !hasOpt(opts, "omitzero")
		if r, ok := f.Tag.Lookup("required"); ok {
			required = r == "true"
		}
		if required {
			s.Required = append(s.Required, name)
		}
		*props = append(*props, schemaProp{name: name, schema: fs})
	}
}

func hasOpt(opts, want string) bool {
	for _, o := range strings.Split(opts, ",") {
		if o == want {
			return true
		}
	}
	return false
}

func parseEnum(tag, typ string) []any {
	var out []any
	for _, v := range strings.Split(tag, ",") {
		v = strings.TrimSpace(v)
		switch typ {
		case "integer":
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				out = append(out, n)
				continue
			}
		case "number":
			if n, err := strconv.ParseFloat(v, 64); err == nil {
				out = append(out, n)
				continue
			}
		case "boolean":
			if b, err := strconv.ParseBool(v); err == nil {
				out = append(out, b)
				continue
			}
		}
		out = append(out, v)
	}
	return out
}
