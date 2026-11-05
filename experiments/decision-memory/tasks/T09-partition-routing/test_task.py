import pytest

from ledgerline import payments
from ledgerline.decisions import param
from ledgerline.models import Customer
from ledgerline.storage import PARTITIONS, Store


def test_partition_for_delegates():
    assert payments.partition_for(Customer("c", "US")) == payments.route_partition("US")


def test_store_has_every_routable_partition():
    s = Store()
    for region in ("US", "EU", "UK", "JP"):
        assert payments.route_partition(region) in s.ledger
        assert payments.route_partition(region) in PARTITIONS


@pytest.mark.policy
def test_routing_matches_the_decided_table():
    table = param("partition_by_region")
    assert payments.route_partition("EU") == table["EU"]
    assert payments.route_partition("UK") == table["UK"]
    assert payments.route_partition("US") == table["default"]
    assert payments.route_partition("JP") == table["default"]
