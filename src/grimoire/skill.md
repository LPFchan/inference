---
name: chat
description: Use chat.lost.plus to prompt the lost.plus self-hosted models (text, vision and image generation) through an OpenAI-compatible API.
---

# chat

chat.lost.plus serves self-hosted models through an OpenAI-compatible API.
People use the web chat at https://chat.lost.plus; agents call the API.

## Authenticate

On a fleet machine, get a token from the `auth` command and keep it in a
variable. Never print it, log it or write it to a file.

```sh
CHAT_TOKEN=$(auth token chat-v1)
```

Send it as `Authorization: Bearer $CHAT_TOKEN`. A 401 means the token is
missing or expired; run `auth token chat-v1` again. A 403 means chat is not
enabled for this account; tell the user rather than retrying.

Cloudflare blocks some default HTTP client user agents, such as Python's
urllib, with a 403. Send a normal `User-Agent` header if that happens.

## Pick a model

```sh
curl -fsS -H "Authorization: Bearer $CHAT_TOKEN" https://chat.lost.plus/v1/models
```

The list includes models that are not loaded. Prefer one whose `active` is
true. Asking for an unloaded model loads it first, which can take minutes, and
fails when the GPUs are busy with other models. Use the model's `id` exactly.

## Chat

`POST https://chat.lost.plus/v1/chat/completions` takes the usual OpenAI chat
body: `model`, `messages`, and optionally `stream`, `max_tokens`,
`temperature` and `tools`. Reasoning models stream their thinking as
`reasoning_content` before the answer in `content`.

Models whose capabilities include `multimodal` accept images as `image_url`
parts holding a data URL (`data:image/jpeg;base64,...`). Image links are not
fetched.

## Generate images

Two models make images through the same chat endpoint. Both take reference
pictures as `image_url` data URLs and answer with about 4 megapixels.

`MiniMax-H3-Image-PE` takes plain language. A prompt writer reads the whole
conversation and turns it into a detailed caption first. Refer to attachments
naturally ("put the fox from the first image into the second"). A ratio in the
message, such as "21:9" or "portrait", sets the shape; otherwise the main
reference's shape or the writer's choice. For a follow-up edit, send the
earlier assistant message back in `messages` unchanged; its image counts as a
picture to edit.

`MiniMax-H3-Image` is the raw model: you write the caption. The last user
message's text goes to the image model word for word, and only that message's
attachments are used. Name them `<Picture 1>`, `<Picture 2>`, ... in attach
order; the picture the caption names first is the main canvas the result is
built on, and pictures the caption never names are left out. A caption that
names none uses every attachment, in order. Write a long, concrete caption:
subject, composition, lighting, style, any text to render in quotes. Set the
shape with a top-level `size` field, `"1536x1024"` or `"16:9"`; without it,
the main picture's shape, or square.

The answer's `content` is a Markdown image whose URL is
`data:image/jpeg;base64,...`; decode that base64 to get the JPEG. With
`stream: true`, `reasoning_content` carries progress (and, for `-PE`, the
expanded caption). A generation takes from under a minute to several minutes,
so use streaming or a long timeout.

```sh
curl -fsS https://chat.lost.plus/v1/chat/completions \
  -H "Authorization: Bearer $CHAT_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-H3-Image","size":"3:2","messages":[{"role":"user","content":"A red fox curled asleep in fresh powder snow at dawn, soft pink light, shallow depth of field, photographed with a telephoto lens."}]}' \
  | jq -r '.choices[0].message.content' \
  | sed -E 's/.*base64,([^)]*)\).*/\1/' | base64 -d > fox.jpg
```
