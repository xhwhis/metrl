import httpx
from conftest import Cluster, wait_for


def test_keys_and_workers_follow_postgres(cluster: Cluster) -> None:
    # The server is already running; the new key reaches it only through CDC.
    key = cluster.key("late")
    assert cluster.capabilities("tml-unknown").status_code == 401
    assert cluster.capabilities(key).json()["supported_models"] == [
        {"model_name": "mock", "max_context_length": 4096, "trainable": True, "sampleable": True}
    ]

    # Changing a primary key in place must revoke the old value too.
    rotated = f"{key}-rotated"
    cluster.psql("rotate_api_key", old=key, new=rotated)
    wait_for(
        lambda: cluster.capabilities(rotated).status_code == 200,
        "a rotated key was never accepted",
    )
    assert cluster.capabilities(key).status_code == 401
    key = rotated

    cluster.worker.terminate()
    assert cluster.worker.wait(timeout=10) == 0, "the worker must shut down cleanly"
    # Unregistering on shutdown removes the base without waiting for a timeout.
    wait_for(
        lambda: not cluster.capabilities(key).json()["supported_models"],
        "a stopped worker stayed registered",
    )
    session = httpx.post(
        f"{cluster.url}/api/v1/create_session", headers={"x-api-key": key}, timeout=2
    ).json()["session_id"]
    response = httpx.post(
        f"{cluster.url}/api/v1/create_sampling_session",
        headers={"x-api-key": key},
        json={"session_id": session, "sampling_session_seq_id": 0, "base_model": "mock"},
        timeout=2,
    )
    assert response.status_code == 400


def test_view_survives_a_lost_stream(cluster: Cluster) -> None:
    # The worker registered while the server was streaming, so after the stream
    # is lost only the resync scan keeps it in the view.
    assert cluster.psql("terminate_walsender") == "t", "the server must stream from one slot"
    key = cluster.key("after-resync")
    assert cluster.capabilities(key).json()["supported_models"], "the resync scan lost the worker"


def test_slot_releases_wal_of_unpublished_writes(cluster: Cluster) -> None:
    # Writes to unpublished tables reach the server only as keepalives; if those
    # are not acked, the slot pins WAL until the disk fills.
    cluster.psql("insert_org", org="unpublished")
    lsn = cluster.psql("wal_insert_lsn")
    # The server sends acks every second.
    wait_for(
        lambda: cluster.psql("slot_confirmed", lsn=lsn) == "t", f"the slot never confirmed {lsn}"
    )
