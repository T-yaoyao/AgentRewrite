"""
SQL Fingerprint Generator
Generates normalized SQL templates for similarity matching
"""

import re
from typing import Optional

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


class SQLFingerprintGenerator:
    """Generate normalized SQL fingerprints for similarity matching"""
    
    def __init__(self):
        self.placeholder_patterns = [
            (r'\b\d+\.?\d*\b', '?'),  # Numbers
            (r"'[^']*'", '?'),  # Single-quoted strings
            (r'"[^"]*"', '?'),  # Double-quoted strings
        ]
    
    def get_template(self, sql: str) -> str:
        """
        Generate normalized SQL template (fingerprint)
        
        Args:
            sql: Original SQL query
            
        Returns:
            Normalized SQL template string
        """
        if not sql:
            return ""
        
        # Step 1: Remove comments
        sql = self._remove_comments(sql)
        
        # Step 2: Normalize whitespace
        sql = self._normalize_whitespace(sql)
        
        # Step 3: Parameterize constants
        sql = self._parameterize_constants(sql)
        
        # Step 4: Normalize case (keywords uppercase, identifiers lowercase)
        sql = self._normalize_case(sql)
        
        # Step 5: Remove extra whitespace again
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def _remove_comments(self, sql: str) -> str:
        """Remove SQL comments"""
        # Remove single-line comments
        sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        # Remove multi-line comments
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql
    
    def _normalize_whitespace(self, sql: str) -> str:
        """Normalize whitespace"""
        # Replace all whitespace with single space
        sql = re.sub(r'\s+', ' ', sql)
        # Remove spaces around operators and punctuation
        sql = re.sub(r'\s*([=<>!+\-*/(),;])\s*', r'\1', sql)
        # Add space after keywords
        sql = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|ON|GROUP|ORDER|HAVING|UNION|INSERT|UPDATE|DELETE)\b', r' \1 ', sql, flags=re.IGNORECASE)
        return sql.strip()
    
    def _parameterize_constants(self, sql: str) -> str:
        """Replace constants with placeholders"""
        # Replace numbers
        sql = re.sub(r'\b\d+\.?\d*\b', '?', sql)
        # Replace single-quoted strings
        sql = re.sub(r"'[^']*'", '?', sql)
        # Replace double-quoted strings (but keep table/column names in quotes if they are identifiers)
        # We'll be more careful here - only replace string literals, not quoted identifiers
        # For now, replace all double-quoted content
        sql = re.sub(r'"[^"]*"', '?', sql)
        return sql
    
    def _normalize_case(self, sql: str) -> str:
        """Normalize SQL case - keywords uppercase, rest lowercase"""
        if SQLPARSE_AVAILABLE:
            try:
                # Use sqlparse to properly handle SQL structure
                parsed = sqlparse.parse(sql)
                if not parsed:
                    return sql.upper()
                
                # Reconstruct with normalized case
                normalized_parts = []
                for statement in parsed:
                    for token in statement.flatten():
                        if token.ttype is None or token.ttype in sqlparse.sql.Keyword:
                            # SQL keywords - uppercase
                            normalized_parts.append(token.value.upper())
                        elif token.ttype in sqlparse.sql.String:
                            # String literals - keep as placeholder
                            normalized_parts.append('?')
                        elif token.ttype in sqlparse.sql.Number:
                            # Numbers - keep as placeholder
                            normalized_parts.append('?')
                        else:
                            # Other tokens - lowercase
                            normalized_parts.append(token.value.lower())
                
                return ' '.join(normalized_parts)
            except Exception as e:
                # Fallback to simple normalization
                pass
        
        # Fallback: simple case normalization
        # Keywords to uppercase
        keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 
                   'OUTER', 'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'UNION', 'ALL', 
                   'DISTINCT', 'AS', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIMIT', 'OFFSET']
        sql_upper = sql.upper()
        for keyword in keywords:
            sql_upper = re.sub(rf'\b{keyword}\b', keyword, sql_upper, flags=re.IGNORECASE)
        return sql_upper

