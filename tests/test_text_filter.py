from src.data.text_filter import evaluate_text_filter


def test_en_web_quality_filter_accepts_educational_math_text():
    text = 2 * (
        "A useful way to understand the derivative is to compare the average rate of change "
        "on a small interval with the limiting rate at a single point. In a classroom example, "
        "let f(x) = x^2 + 3x. The slope between x and x + h is ((x + h)^2 + 3(x + h) - x^2 - 3x) / h, "
        "which simplifies to 2x + h + 3. As h approaches zero, the derivative becomes 2x + 3. "
        "This calculation explains why a tangent line gives the best local linear approximation, "
        "and it also shows how symbolic rules are connected to numerical change. "
    )

    result = evaluate_text_filter(text, {"preset": "en_web_quality_v1"})

    assert result.accepted
    assert result.reasons == ()


def test_en_web_quality_filter_rejects_product_cookie_error_boilerplate():
    text = """
    Cookie Preferences
    Accept cookies
    Privacy Policy
    Terms of Service
    Sign in
    Main menu
    Error 404
    Page not found
    Product details
    Customer reviews
    Add to cart
    Checkout
    SKU: 123-456-789
    Price: $19.99
    Shipping
    Returns
    https://example.com/product
    https://example.com/cart
    https://example.com/checkout
    https://example.com/privacy
    https://example.com/error
    Broken text: � � �
    """

    result = evaluate_text_filter(text, {"preset": "en_web_quality_v1"})

    assert not result.accepted
    assert "boilerplate" in result.reasons
    assert "too_many_urls" in result.reasons
    assert "suspicious_sequences" in result.reasons
