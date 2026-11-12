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

`MiniMax-H3-Image` makes images through the same chat endpoint. Describe the
picture in plain language. To edit or combine pictures, attach them as
`image_url` data URLs and refer to them in the message ("put the fox from the
first image into the second"). A ratio in the message, such as "21:9" or
"portrait", sets the output shape; otherwise the main reference's shape or the
model's own choice is used. Outputs are about 4 megapixels.

The answer's `content` is a Markdown image whose URL is
`data:image/jpeg;base64,...`; decode that base64 to get the JPEG. With
`stream: true`, `reasoning_content` carries the expanded prompt and progress.
A generation takes from under a minute to several minutes, so use streaming or
a long timeout. For a follow-up edit, send the earlier assistant message back
in `messages` unchanged; its image counts as a picture to edit.

```sh
curl -fsS https://chat.lost.plus/v1/chat/completions \
  -H "Authorization: Bearer $CHAT_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-H3-Image","messages":[{"role":"user","content":"a red fox asleep in fresh snow, 3:2"}]}' \
  | jq -r '.choices[0].message.content' \
  | sed -E 's/.*base64,([^)]*)\).*/\1/' | base64 -d > fox.jpg
```
