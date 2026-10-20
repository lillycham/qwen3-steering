"""Cutting articles into passages and splitting them."""
import json

import numpy as np

from proprio import corpus as C

PROSE = ("The kettle is heated by a coil in its base. When the water boils, steam reaches a small bimetal strip. "
         "The strip bends and trips a switch that cuts the current. This is why a kettle turns itself off. "
         "Most kettles also have a second, safety switch that acts if the first one fails. Older designs relied "
         "on the user to notice the steam. A modern jug can also keep the water warm for several minutes.")


def test_sentences_split_after_a_full_stop_followed_by_a_capital():
    s = C.sentences("It rose. Then it fell! Was that all? yes, 3.5 m. Done.\n\nNext.")
    assert s[:3] == ["It rose.", "Then it fell!", "Was that all? yes, 3.5 m."] and s[-1] == "Next."


def test_an_initial_does_not_end_a_sentence():
    s = C.sentences("The novel was written by H. G. Wells in 1898. It was a hit. J. R. R. Tolkien came later.")
    assert s == ["The novel was written by H. G. Wells in 1898.", "It was a hit.", "J. R. R. Tolkien came later."]


def test_clean_chunks_pass_and_debris_is_dropped():
    assert C.clean_ok(PROSE)
    assert not C.clean_ok("Accipiter () is a genus of birds of prey in the family Accipitridae.")
    assert not C.clean_ok("The acre ( ) is a unit of land area used in the British imperial system.")
    assert not C.clean_ok("It was a town in France  with a large church.")                       # (a double space: a cut-out)
    assert not C.clean_ok("A list: {{cite web}} of things and == Heading == after it.")
    assert not C.clean_ok("See https://example.org for a table | a | b |.")
    assert not C.clean_ok("Il a trouvé des éléments dans la forêt à côté du château ancien.")
    assert not C.clean_ok("In 1998 there were 1234 of 5678 people, 9012 of 3456 homes and 7890 of 1234 farms.")
    assert not C.clean_ok("")


def test_chunks_are_whole_sentences_of_the_right_length_and_deterministic():
    text = " ".join([PROSE] * 6)
    a = C.chunks_of(text, np.random.default_rng(0), per_article=3)
    b = C.chunks_of(text, np.random.default_rng(0), per_article=3)
    assert a == b and 1 <= len(a) <= 3
    for ch in a:
        assert C.LO <= C.words(ch) <= C.HI and ch[0].isupper() and ch.endswith(".")
        assert ch in text.replace("\n", " ")
    assert C.chunks_of("", np.random.default_rng(0)) == [] and C.chunks_of("Too short.", np.random.default_rng(0)) == []


def test_split_is_stable_and_about_70_10_20():
    keys = [f"article {i}" for i in range(3000)]
    splits = [C.split_of(k) for k in keys]
    assert splits == [C.split_of(k) for k in keys]
    share = {s: splits.count(s) / len(keys) for s in ("train", "val", "test")}
    assert abs(share["train"] - 0.7) < 0.03 and abs(share["test"] - 0.2) < 0.03


def write_articles(tmp_path, n=40):
    lines = [json.dumps({"title": f"Kettle {i}", "topic": ["Technology", "Arts"][i % 2], "text": " ".join([PROSE] * 5)})
             for i in range(n)]
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(lines))
    return str(p)


def test_build_writes_the_passage_file_split_by_article(tmp_path, capsys):
    out = str(tmp_path / "wiki.json")
    C.build(write_articles(tmp_path), out, max_chunks=500, per_article=2, seed=0, split_by="title")
    data = json.loads(open(out).read())
    ps = data["passages"]
    assert data["split_by"] == "title" and ps and all({"text", "topic", "category", "split", "source"} <= set(p) for p in ps)
    by_article: dict[str, set] = {}
    for p in ps:
        by_article.setdefault(p["topic"], set()).add(p["split"])
    assert all(len(s) == 1 for s in by_article.values())                       # (an article is in one split only)
    assert {"train", "test"} <= {p["split"] for p in ps}
    assert "chunks from 40 articles" in capsys.readouterr().out


def test_split_by_category_keeps_whole_categories_apart_and_max_chunks_is_respected(tmp_path):
    out = str(tmp_path / "wiki.json")
    C.build(write_articles(tmp_path), out, max_chunks=7, per_article=3, seed=1, split_by="category")
    ps = json.loads(open(out).read())["passages"]
    assert len(ps) == 7
    cats: dict[str, set] = {}
    for p in ps:
        cats.setdefault(p["category"], set()).add(p["split"])
    assert all(len(s) == 1 for s in cats.values())
