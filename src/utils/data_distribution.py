import json

# Database statistics storage
# Format: {database_name: [["table_name", "row_count"], ...]}
DATABASE_STATISTICS = {
    "calcite": [["bonus", "45000000"], ["dept", "450000"], ["emp", "45000000"], ["emp_b", "45000000"], ["empnullables", "45000000"], ["empnullables_20", "55"]],
    
    "tpch": [["customer", "1500000"], ["lineitem", "59986052"], ["nation", "25"], ["orders", "15000000"], ["part", "2000000"], ["partsupp", "8000000"], ["region", "5"], ["supplier", "100000"]],
    
    "dsb": [["call_center", "24"], ["catalog_page", "12000"], ["catalog_returns", "2158260"], ["catalog_sales", "14397492"], ["customer", "500000"], ["customer_address", "250000"], ["customer_demographics", "1920800"], ["date_dim", "73049"], ["dbgen_version", "0"], ["household_demographics", "7200"], ["income_band", "20"], ["inventory", "133110000"], ["item", "102000"], ["promotion", "500"], ["reason", "45"], ["ship_mode", "20"], ["store", "102"], ["store_returns", "7198194"], ["store_sales", "28800991"], ["time_dim", "86400"], ["warehouse", "10"], ["web_page", "200"], ["web_returns", "1440354"], ["web_sales", "7197566"], ["web_site", "42"]],
    
    "imdb_job": [["aka_name", "901343"], ["aka_title", "361472"], ["cast_info", "36244344"], ["char_name", "3140339"], ["comp_cast_type", "4"], ["company_name", "234997"], ["company_type", "4"], ["complete_cast", "135086"], ["info_type", "113"], ["keyword", "134170"], ["kind_type", "7"], ["link_type", "18"], ["movie_companies", "2609129"], ["movie_info", "14835720"], ["movie_info_idx", "1380035"], ["movie_keyword", "4523930"], ["movie_link", "29997"], ["name", "4167491"], ["person_info", "2963664"], ["role_type", "12"], ["title", "2528312"]]
}

def get_data_statistics(db_name: str) -> str:
    statistics = DATABASE_STATISTICS.get(db_name, [])
    return json.dumps(statistics)

def get_statistics_list(db_name: str) -> list:
    return DATABASE_STATISTICS.get(db_name, [])

def update_statistics(db_name: str, statistics: list):
    """
    Update statistics for a database.
    
    Args:
        db_name: Database name
        statistics: List of [table_name, row_count] pairs
    """
    DATABASE_STATISTICS[db_name] = statistics
    print(f"Updated statistics for database '{db_name}' with {len(statistics)} tables")

def get_available_databases() -> list:
    """Get list of available database names."""
    return list(DATABASE_STATISTICS.keys())


if __name__ == "__main__":
    # Demo usage
    print("Available databases:", get_available_databases())
    print("TPC-H S10 statistics:", get_data_statistics("tpch_s10"))