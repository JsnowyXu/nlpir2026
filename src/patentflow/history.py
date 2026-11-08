"""Bounded causal collaboration state and batched temporal history helpers."""
from collections import defaultdict


class RecentCollaborators:
    def __init__(self, limit=3):
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        self.limit = limit
        self.peers = {}
        self.last_timestamp = None

    def snapshot(self, org):
        return tuple(self.peers.get(org, ()))

    def observe_timestamp(self, timestamp, patent_organizations):
        """Call only AFTER reading/predicting all events at this timestamp.

        The argument must contain every coauthored patent at the timestamp;
        equal-time order cannot change memory or create within-day history.
        """
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Timestamp groups must be strictly increasing")
        new = defaultdict(set)
        for orgs in patent_organizations:
            orgs = set(orgs)
            for org in orgs:
                new[org].update(orgs - {org})
        for org, peers in new.items():
            merged = dict(self.peers.get(org, ()))
            merged.update({peer: timestamp for peer in peers})
            self.peers[org] = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[:self.limit]
        self.last_timestamp = timestamp


def timestamp_groups(events, key=lambda row: row["timestamp"]):
    group, previous = [], None
    for row in events:
        stamp = key(row)
        if previous is not None and stamp < previous:
            raise ValueError("Events must be sorted by timestamp")
        if previous is not None and stamp != previous:
            yield previous, group
            group = []
        group.append(row)
        previous = stamp
    if group:
        yield previous, group
