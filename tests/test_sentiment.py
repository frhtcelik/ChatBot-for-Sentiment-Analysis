from app.sentiment import analyze_sentiment, get_word_sentiments


def test_positive_turkish_sentence():
    result = analyze_sentiment("Bugun cok mutluyum, proje harika ilerliyor.")

    assert result.label == "positive"
    assert result.score > 0


def test_negative_turkish_sentence():
    result = analyze_sentiment("Bu deneyim berbat ve kotu oldu.")

    assert result.label == "negative"
    assert result.score < 0


def test_empty_text_is_neutral():
    result = analyze_sentiment("   ")

    assert result.label == "neutral"
    assert result.confidence == 0.0


def test_ml_model_prediction():
    result = analyze_sentiment("Bugün harika bir gün.", model_type="ml")
    assert result.label == "positive"
    assert result.score > 0
    assert result.confidence > 0.4

    result_neg = analyze_sentiment("Bu hiç iyi olmadı, çok kötü.", model_type="ml")
    assert result_neg.label == "negative"
    assert result_neg.score < 0


def test_get_word_sentiments():
    # Test hybrid
    res_hybrid = get_word_sentiments("bugün harika bir gün", model_type="hybrid")
    scores_hybrid = {w: s for w, s in res_hybrid}
    assert scores_hybrid.get("harika", 0.0) > 0.0

    # Test ml
    res_ml = get_word_sentiments("bugün çok kötü", model_type="ml")
    scores_ml = {w: s for w, s in res_ml}
    assert scores_ml.get("kötü", 0.0) < 0.0


