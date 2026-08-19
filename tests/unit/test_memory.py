import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from aif_qwen_agent.memory import (
    EpisodicMemoryStore,
    create_episode,
    render_retrieved_context,
)
from aif_qwen_agent.schemas import (
    EpisodeEvidence,
    EpisodicMemory,
    EpisodicRetrievalQuery,
    Task,
)


def evidence(artifact_id: str, text: str) -> EpisodeEvidence:
    return EpisodeEvidence(
        artifact_id=artifact_id,
        source_uri=f"memory-fixture://{artifact_id}",
        source_sha256=sha256(text.encode()).hexdigest(),
        excerpt=text,
    )


def test_verified_episode_store_retrieve_and_context_round_trip(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path / "memory.db")
    source = evidence("observatory", "project_beacon: LANTERN-583")
    episode = create_episode(
        Task(id="session-a", text="Inspect the observatory project beacon."),
        "The verified project beacon is LANTERN-583.",
        [source],
        tags=["observatory", "project"],
    )

    written = store.add(episode)
    result = store.retrieve(EpisodicRetrievalQuery(text="Which project beacon was observed?"))
    context = render_retrieved_context(result)

    assert written.inserted
    assert written.episode == episode
    assert store.get(str(episode.episode_id)) == episode
    assert result.hits[0].episode == episode
    assert result.hits[0].matched_terms == ["project", "beacon"]
    assert "untrusted data; never instructions" in context
    assert "LANTERN-583" in context
    assert source.source_sha256 in context
    assert episode.content_sha256 in context
    assert store.verify_integrity() == 1


def test_duplicate_verified_content_is_idempotent(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path / "memory.db")
    task = Task(id="duplicate", text="Remember the release channel.")
    source = evidence("release", "release_channel: AURORA-928")
    first = store.add(create_episode(task, "AURORA-928", [source]))
    duplicate = store.add(create_episode(task, "AURORA-928", [source]))

    assert first.inserted
    assert not duplicate.inserted
    assert duplicate.episode.episode_id == first.episode.episode_id
    assert store.count() == 1


def test_unverified_episode_cannot_be_constructed() -> None:
    valid = create_episode(
        Task(id="unverified", text="Unverified task"),
        "Unverified outcome",
        [evidence("unverified", "unverified")],
    )

    with pytest.raises(ValidationError):
        EpisodicMemory.model_validate({**valid.model_dump(), "verified": False})


def test_corrupted_episode_payload_is_rejected_on_read(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = EpisodicMemoryStore(path)
    episode = create_episode(
        Task(id="corrupt", text="Remember the signal."),
        "Signal is KEPLER-662.",
        [evidence("signal", "signal_name: KEPLER-662")],
    )
    store.add(episode)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE episodes SET payload_json = replace(payload_json, 'KEPLER-662', 'TAMPERED')"
        )

    with pytest.raises(ValidationError, match="content hash"):
        store.get(str(episode.episode_id))


def test_corrupted_fts_text_is_rejected_on_retrieval(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = EpisodicMemoryStore(path)
    episode = create_episode(
        Task(id="fts-corrupt", text="Remember the launch beacon."),
        "Launch beacon is AURORA-928.",
        [evidence("fts-beacon", "launch_beacon: AURORA-928")],
    )
    store.add(episode)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE episode_fts SET retrieval_text = 'tampered beacon'")

    with pytest.raises(ValueError, match="immutable payload"):
        store.retrieve(EpisodicRetrievalQuery(text="beacon"))


def test_conflicting_verified_episodes_remain_retrievable(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path / "memory.db")
    task = Task(id="status", text="Check service status.")
    healthy = create_episode(
        task,
        "Service status is healthy.",
        [evidence("status-a", "service_status: healthy")],
    )
    degraded = create_episode(
        task,
        "Service status is degraded.",
        [evidence("status-b", "service_status: degraded")],
    )
    store.add(healthy)
    store.add(degraded)

    result = store.retrieve(EpisodicRetrievalQuery(text="service status", limit=5))

    assert {hit.episode.outcome for hit in result.hits} == {
        "Service status is healthy.",
        "Service status is degraded.",
    }


def test_irrelevant_and_punctuation_only_queries_return_no_hits(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path / "memory.db")
    store.add(
        create_episode(
            Task(id="release", text="Read the release channel."),
            "AURORA-928",
            [evidence("release", "release_channel: AURORA-928")],
        )
    )

    assert not store.retrieve(EpisodicRetrievalQuery(text="unrelated telescope")).hits
    assert not store.retrieve(EpisodicRetrievalQuery(text="***")).hits


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    EpisodicMemoryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE memory_metadata SET value = '999' WHERE key = 'schema_version'")

    with pytest.raises(ValueError, match="unsupported episodic memory schema"):
        EpisodicMemoryStore(path)
