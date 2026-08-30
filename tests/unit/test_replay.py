from kwola.training.replay import ReplaySampler


def test_replay_visits_every_trace_before_repeating() -> None:
    sampler = ReplaySampler(10, batch_size=2, world_size=1, rank=0, seed=7, training_step=0)

    indexes = tuple(index for iteration in range(5) for index in sampler.batch_indexes(iteration))

    assert sorted(indexes) == list(range(10))


def test_replay_is_deterministic_but_changes_between_training_steps() -> None:
    first = ReplaySampler(20, 4, 1, 0, seed=8, training_step=2).batch_indexes(0)
    repeated = ReplaySampler(20, 4, 1, 0, seed=8, training_step=2).batch_indexes(0)
    following = ReplaySampler(20, 4, 1, 0, seed=8, training_step=3).batch_indexes(0)

    assert first == repeated
    assert first != following


def test_replay_gives_distributed_ranks_disjoint_slices() -> None:
    rank_zero = ReplaySampler(24, 4, 2, 0, seed=9, training_step=1)
    rank_one = ReplaySampler(24, 4, 2, 1, seed=9, training_step=1)

    assert set(rank_zero.batch_indexes(0)).isdisjoint(rank_one.batch_indexes(0))
    combined = rank_zero.batch_indexes(0) + rank_one.batch_indexes(0)
    assert len(set(combined)) == 8


def test_distributed_slices_stay_disjoint_across_epoch_boundaries() -> None:
    rank_zero = ReplaySampler(10, 4, 2, 0, seed=9, training_step=1)
    rank_one = ReplaySampler(10, 4, 2, 1, seed=9, training_step=1)

    for iteration in range(4):
        assert set(rank_zero.batch_indexes(iteration)).isdisjoint(rank_one.batch_indexes(iteration))


def test_small_replay_cycles_with_a_new_epoch_permutation() -> None:
    sampler = ReplaySampler(3, 4, 1, 0, seed=10, training_step=0)

    indexes = sampler.batch_indexes(0)

    assert sorted(indexes[:3]) == [0, 1, 2]
    assert indexes[3] in {0, 1, 2}
