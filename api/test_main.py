# test_main.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

test_exp = "It's no use going back to yesterday, because I was a different person then"
label_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "text_examples": [test_exp]
    }
    print("TEST 1 PASSED")


def test_single_essay_scoring():
    response = client.post(
        "http://0.0.0.0:8080/single_essay",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        json={
            "essay": test_exp
        }

    )
    assert response.status_code == 200
    response = response.json()
    assert response["text"] == test_exp
    # check all analysis metrics are in response
    assert len(list(map(lambda x: x in response.keys(), label_cols))) == len(label_cols)
    # check all analysis metrics are in response
    assert len(
        list(
            map(
                lambda x: (response[x] < 5) and (response[x] > 0),
                label_cols
            )
        )
    ) == len(label_cols)
    print("TEST 2 PASSED")


def test_multiple_essays_scoring():
    N = 3
    response = client.post(
        "http://0.0.0.0:8080/multiple_essays",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        json={
            "essays": [test_exp] * N
        }

    )
    assert response.status_code == 200
    response = response.json()
    assert len(response["batch"]) == N
    for element in response["batch"]:
        assert element["text"] == test_exp
        # check all analysis metrics are in response
        assert len(list(map(lambda x: x in element.keys(), label_cols))) == len(label_cols)
        # check all analysis metrics are in response
        assert len(list(map(lambda x: (element[x] < 5) and (element[x] > 0), label_cols))) == len(label_cols)
    print("TEST 3 PASSED")