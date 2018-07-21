from sqlalchemy import text

from db_utils import get_connection
from kafka_config import DB_CONFIG


def insert_classes(content_id, scored_classes):
    # TODO enforce unique content_id, class pairs
    social_db = get_connection(DB_CONFIG)
    values = [str((str(content_id), sc['class'], str(sc['score']))) for sc in scored_classes]
    query = text('insert into social.concepts (content_id, concept, score) \
                    values' + ', '.join(values) + ';')
    social_db.execute(query)
