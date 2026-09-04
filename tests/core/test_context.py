from enumeraite.core.context import collect_path_context, collect_subdomain_context


def test_path_context_finds_target_conventions():
    bundle = collect_path_context(
        [
            "/api/v1/user-profiles/123",
            "/api/v2/user-orders/456",
            "/api/v1/user-profiles/789",
        ]
    )
    rendered = bundle.render()
    assert "API versioning" in rendered
    assert "kebab" in rendered
    assert "common path prefix: /api" in rendered
    assert "identifiers" in rendered


def test_subdomain_context_finds_environment_and_instances():
    rendered = collect_subdomain_context(
        [
            "api-dev-01.example.com",
            "api-prod-02.example.com",
            "api-stage-03.example.com",
        ]
    ).render()
    assert "shared apex: example.com" in rendered
    assert "environment tiers" in rendered
    assert "numbered host instances" in rendered


def test_context_needs_two_items():
    assert collect_path_context(["/only"]).render() == ""
