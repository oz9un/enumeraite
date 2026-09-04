from enumeraite.core.candidate_pool import structural_candidates


def test_path_structural_candidates_fill_parents_and_version_grid():
    known = [
        "/api/v1/users/{userId}/profile",
        "/api/v1/products/search",
        "/api/v2/orders",
    ]
    values = [item.value for item in structural_candidates(known, "path", count=50)]
    assert "/api/v1/users/123" in values
    assert "/api/v1/products" in values
    assert "/api/v1/orders" in values
    assert "/api/v2/users" in values
    assert "/api/v2/products" in values


def test_subdomain_structural_candidates_fill_observed_environment_grid():
    known = ["api-dev.example.com", "cdn-prod.example.com"]
    values = [
        item.value for item in structural_candidates(known, "subdomain", "example.com", count=20)
    ]
    assert "api-prod.example.com" in values
    assert "cdn-dev.example.com" in values


def test_subdomain_structural_candidates_find_numbered_suffixes_without_zero_guess():
    known = ["worker-dev-01.example.com", "worker-prod-01.example.com"]
    values = [
        item.value for item in structural_candidates(known, "subdomain", "example.com", count=20)
    ]
    assert "worker-dev-02.example.com" in values
    assert "worker-prod-02.example.com" in values
    assert "worker-dev-00.example.com" not in values


def test_subdomain_structural_candidates_ignore_cross_apex_evidence():
    values = [
        item.value
        for item in structural_candidates(
            ["api-dev.example.com", "worker-01.attacker.test"],
            "subdomain",
            "example.com",
            count=20,
        )
    ]
    assert all(value.endswith(".example.com") for value in values)
    assert all("attacker" not in value and "worker" not in value for value in values)

    try:
        structural_candidates(["api.example.com"], "subdomain", "https://example.com")
    except ValueError as exc:
        assert "valid DNS apex" in str(exc)
    else:
        raise AssertionError("invalid structural target was accepted")


def test_structural_candidates_materialize_generators_and_honor_zero_budget():
    assert structural_candidates(iter(["/api/v1/users/profile"]), "path", count=0) == []
    known = (value for value in ["/api/v1/users", "/api/v2/orders"])
    values = [item.value for item in structural_candidates(known, "path", count=20)]
    assert "/api/v1/orders" in values
    assert "/api/v2/users" in values
    assert "/api/v1/users" not in values
    try:
        structural_candidates(["/seed"], "path", count=5001)
    except ValueError as exc:
        assert "between 0 and 5000" in str(exc)
    else:
        raise AssertionError("oversized structural budget was accepted")
