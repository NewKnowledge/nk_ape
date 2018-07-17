import os

KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'posts')
KAFKA_BROKERS = os.getenv('KAFKA_BROKERS', 'kafka:9092')

SOCIAL_DB_HOST = os.getenv('SOCIAL_DB_HOST', 'social-db')
SOCIAL_DB_NAME = os.getenv('SOCIAL_DB_NAME', 'social')
SOCIAL_DB_USER = os.getenv('SOCIAL_DB_USER', 'social')
SOCIAL_DB_PASS = os.getenv('SOCIAL_DB_PASS', '')

API_PASSWORD = os.getenv('API_PASSWORD', 'pizza')

DB_CONFIG = {
    'host': SOCIAL_DB_HOST,
    'db_name': SOCIAL_DB_NAME,
    'user': SOCIAL_DB_USER,
    'password': SOCIAL_DB_PASS
}
