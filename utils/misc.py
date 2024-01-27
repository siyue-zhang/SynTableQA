import sqlite3

def fetch_table_data(connection, table_name):
    cursor = connection.execute(f"SELECT * FROM {table_name}")
    header = [col[0] for col in cursor.description]
    rows = [list(map(str, row)) for row in cursor.fetchall()]
    return {'header': header, 'rows': rows}

def read_sqlite_database(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # Use Row factory to get rows as dictionaries
    connection.text_factory = lambda b: b.decode(errors = 'ignore')

    # Get a list of all tables in the database
    # print('reading database ...')
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    # Fetch data for each table
    database_dict = {}
    for table in tables:
        database_dict[table] = fetch_table_data(connection, table)

    connection.close()
    return database_dict

def execute_query(db_path, query):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    cursor.execute(query)
    result = cursor.fetchall()
    ret = []
    if result:
        for row in result:
            for cell in row:
                ret.append(str(cell))
    connection.close()
    return ret