import asyncio
import base64
import io

import pytest
from PIL import Image

from grimoire import image_server as srv


def _png(width=30, height=20):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_single_request_is_passed_verbatim_with_attachments_numbered():
    a, b = _png(), _png(40, 40)
    pictures, instruction = srv.read_conversation([
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": [
            {"type": "text", "text": "Put the cat from the second picture  into the first."},
            {"type": "image_url", "image_url": {"url": _data_url(a)}},
            {"type": "image_url", "image_url": {"url": _data_url(b)}},
        ]},
    ])
    assert pictures == [a, b]
    assert instruction == "Put the cat from the second picture into the first."


def test_follow_up_turn_sees_earlier_output_as_a_picture():
    ref, out = _png(), _png(64, 36)
    pictures, instruction = srv.read_conversation([
        {"role": "user", "content": [{"type": "text", "text": "a red room"},
                                     {"type": "image_url", "image_url": {"url": _data_url(ref)}}]},
        {"role": "assistant", "content": f"![MiniMax-H3-Image]({_data_url(out)})"},
        {"role": "user", "content": "make it darker"},
    ])
    assert pictures == [ref, out]
    assert "User: a red room [attached <Picture 1>]" in instruction
    assert "Assistant: generated <Picture 2>" in instruction
    assert instruction.endswith("Current request: make it darker")


def test_only_the_newest_sixteen_pictures_are_kept_and_renumbered(monkeypatch):
    monkeypatch.setattr(srv, "MAX_PICTURES", 2)
    imgs = [_png(10 + i, 10) for i in range(3)]
    pictures, instruction = srv.read_conversation([
        {"role": "user", "content": [{"type": "text", "text": "one"},
                                     {"type": "image_url", "image_url": {"url": _data_url(imgs[0])}}]},
        {"role": "assistant", "content": f"![x]({_data_url(imgs[1])})"},
        {"role": "user", "content": [{"type": "text", "text": "two"},
                                     {"type": "image_url", "image_url": {"url": _data_url(imgs[2])}}]},
    ])
    assert pictures == imgs[1:]
    assert "User: one\n" in instruction
    assert "generated <Picture 1>" in instruction
    assert "two [attached <Picture 2>]" in instruction


def test_image_links_are_rejected_instead_of_dropped():
    with pytest.raises(srv.GenerationError, match="uploaded"):
        srv.read_conversation([{"role": "user", "content": [
            {"type": "text", "text": "make this darker"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}])


def test_conversation_without_a_user_turn_is_rejected():
    with pytest.raises(srv.GenerationError):
        srv.read_conversation([{"role": "system", "content": "hi"}])


def test_writer_messages_follow_minimax_edit_assembly():
    messages = srv.writer_messages([_png(30, 20)], "swap the car")
    parts = messages[0]["content"]
    assert parts[1]["text"].endswith("labeled <Picture 1> through <Picture 1>.")
    assert parts[2]["text"] == "Reference Media:\n- <Picture 1> [aspect_ratio 3:2]"
    assert parts[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert '"swap the car"' in parts[4]["text"] and "adaptive" in parts[4]["text"]


def test_writer_messages_use_t2i_prompts_without_pictures():
    messages = srv.writer_messages([], "a fox")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].rstrip().endswith("input_text: a fox\noutput_text:")


def test_parse_writer_answer_accepts_both_ratio_keys_and_stray_text():
    assert srv.parse_writer_answer('ok {"expanded_caption": " A. ", "aspect_ratio": "16:9"}') == ("A.", 16 / 9)
    assert srv.parse_writer_answer('{"expanded_caption": "B", "target_aspect_ratio": "3:2"}') == ("B", 1.5)
    with pytest.raises(srv.GenerationError):
        srv.parse_writer_answer("no json here")


def test_output_size_is_about_four_megapixels_in_multiples_of_32():
    for aspect in (1.0, 16 / 9, 2 / 3, 21 / 9):
        w, h = srv.output_size(aspect)
        assert w % 32 == 0 and h % 32 == 0
        assert abs(w * h - srv.OUTPUT_PIXELS) / srv.OUTPUT_PIXELS < 0.03
        assert abs(w / h - aspect) < 0.03
    assert srv.output_size(10) == srv.output_size(3)


def test_only_named_pictures_reach_h3_renumbered():
    caption, named = srv.select_pictures("Edit <Picture 3> in the style of <Picture 1>, not <Picture 9>. <Picture 3>.", 4)
    assert named == [3, 1]
    assert caption == "Edit <Picture 1> in the style of <Picture 2>, not <Picture 9>. <Picture 1>."


def test_workflow_wires_each_reference_into_conditioning():
    g = srv.build_workflow("cap", 2048, 2048, ["a.png", "b.png"], 7)
    cond = g["4"]["inputs"]
    assert cond["prompt"] == "cap" and (cond["width"], cond["height"]) == (2048, 2048)
    assert cond["images.image_1"] == ["103", 0] and cond["images.image_2"] == ["203", 0]
    assert g["201"]["inputs"]["image"] == "b.png"
    assert "TextGenerateV2" not in {n["class_type"] for n in g.values()}


def test_raw_request_is_the_last_user_message_verbatim():
    a, b = _png(30, 20), _png(20, 30)
    messages = [
        {"role": "user", "content": "an earlier caption"},
        {"role": "assistant", "content": f"![x]({_data_url(_png())})"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(a)}},
            {"type": "text", "text": "  <Picture 2> worn by the fox in <Picture 1>  "},
            {"type": "image_url", "image_url": {"url": _data_url(b)}},
        ]},
    ]
    pictures, caption = srv.read_raw_request(messages)
    assert pictures == [a, b]
    assert caption == "<Picture 2> worn by the fox in <Picture 1>"


def test_raw_request_needs_a_caption():
    with pytest.raises(srv.GenerationError):
        srv.read_raw_request([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(_png())}}]}])


def test_parse_size_accepts_pixels_or_ratio():
    assert srv.parse_size("1536x1024") == 1.5
    assert srv.parse_size("21:9") == 21 / 9
    assert srv.parse_size(None) is None and srv.parse_size("auto") is None
    with pytest.raises(srv.GenerationError):
        srv.parse_size("wide")


def test_raw_model_uses_the_caption_and_main_reference_shape(monkeypatch):
    seen = {}

    async def fake_upload(client, raw):
        return f"pic{len(raw)}"

    async def fake_render(client, workflow):
        seen["workflow"] = workflow
        return b"jpeg"

    async def no_writer(*args):
        raise AssertionError("the raw model must not call the writer")

    monkeypatch.setattr(srv, "upload_picture", fake_upload)
    monkeypatch.setattr(srv, "render", fake_render)
    monkeypatch.setattr(srv, "stream_writer", no_writer)

    async def emit(text):
        pass

    tall = _png(20, 40)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(tall)}},
        {"type": "text", "text": "a fox in snow"},
    ]}]
    content = asyncio.run(srv.generate(srv.RAW_MODEL, messages, None, emit))
    assert content.startswith("![MiniMax-H3-Image](data:image/jpeg;base64,")
    cond = seen["workflow"]["4"]["inputs"]
    assert cond["prompt"] == "a fox in snow"
    assert (cond["width"], cond["height"]) == srv.output_size(0.5)
    assert cond["images.image_1"] == ["103", 0]
    assert seen["workflow"]["101"]["inputs"]["image"] == f"pic{len(tall)}"
