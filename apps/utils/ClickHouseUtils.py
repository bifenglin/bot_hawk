from clickhouse_driver import Client

class ClickHouseUtils:
    def __init__(self, host, user, password, database):
        self.client = Client(host=host, user=user, password=password, database=database)

    def execute_query(self, query):
        return self.client.execute(query)
