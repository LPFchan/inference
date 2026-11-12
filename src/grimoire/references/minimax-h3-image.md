# Writing captions for MiniMax-H3-Image

The raw `MiniMax-H3-Image` model renders its caption exactly as written, and
it does best with the long, concrete English captions MiniMax trained it on.
This page holds MiniMax's official prompt-expansion prompts, the ones
`MiniMax-H3-Image-PE` uses, so you can be the prompt writer yourself or hand
the job to a stronger model.

## The flow

1. Take the user's request and their reference pictures, numbered
   `<Picture 1>`, `<Picture 2>`, ... in the order you will attach them.
2. Follow the prompt below that fits (text to image without pictures, editing
   with them) and produce its JSON answer.
3. Call `MiniMax-H3-Image` with `expanded_caption` as the user message text,
   the same pictures attached in the same order, and the answer's ratio
   (`aspect_ratio` or `target_aspect_ratio`) as `size`.

The caption's `<Picture N>` labels choose which pictures the image model sees,
and the first one named is the canvas it builds on. Pictures the caption never
names are left out.

When you are the writer, read the prompt as your own instructions. When
another model is, send it the messages exactly as laid out below.

## Text to image (no pictures)

System message:

~~~text
{t2i_system}
~~~

User message, with the user's request in place of `<USER_REQUEST>`:

~~~text
{t2i_task}
~~~

The answer is `{{"expanded_caption": "...", "aspect_ratio": "W:H"}}`.

## Editing (one or more pictures)

One user message whose parts are, in order:

1. Text: the task prompt below.
2. Text: `Reference Image Count:\n<N> input image(s), labeled <Picture 1> through <Picture N>.`
3. For each picture: text `Reference Media:\n- <Picture i> [aspect_ratio W:H]`,
   with W:H the picture's own shape reduced to lowest terms, then the picture
   as an `image_url` part. About 1 megapixel is plenty for the writer to see it.
4. Text:
   `\nUser Editing Instruction (JSON string):\n<the request as a JSON string>\n\nUser Aspect Ratio Selection:\nadaptive\n\nApply the instructions above to the supplied inputs and reference images. Output only the required two-field JSON object.`
   Put a concrete ratio such as `16:9` in place of `adaptive` when the user
   asked for one.

The answer is `{{"expanded_caption": "...", "target_aspect_ratio": "W:H"}}`.

Task prompt:

~~~text
{edit_task}
~~~
