import numpy as np
from sqlalchemy import text

from db_utils import get_connection
from kafka_config import DB_CONFIG


# INSERT into social.topics (type, hashtag, topic)
#     select distinct rt.type, rt.hashtag, rt.topic from raw_topics rt where not exists
#         (select st.type, st.hashtag, st.topic from social.topics st where st.hashtag = rt.hashtag or st.topic = rt.topic);

# INSERT into social.posts_topics (topic_Id, post_id)
#     select distinct st.topic_id, sp.post_id from raw_topics rt
#         inner join raw_posts rp on rp.post_id = rt.post_id
#         inner join social.topics st on st.topic = rt.topic
#         inner join social.topics st2 on st2.hashtag = rt.hashtag
#         inner join social.posts sp on sp.content_id = rp.content_id where not exists
#             (select spt.topic_id, spt.post_id from social.posts_topics spt where spt.topic_id = st.topic_id and spt.post_id = sp.post_id);

# def insert_clusters(community_name, urls, labels):
#     connection = get_connection(DB_CONFIG)
#     # image_url, community_name, cluster_label
#     values = [str(tupl) for tupl in zip(urls, [community_name] * len(urls), labels)]
#     query = text('INSERT INTO image_clusters.cluster_labels (image_url, community_name, cluster_label) \
#                     VALUES' + ', '.join(values) + ';')
#     connection.execute(query, community_name=community_name)
#     # TODO need to commit after insert?


# def remove_community_clusters(community_name):
#     connection = get_connection(DB_CONFIG['cluster'])
#     query = text('''delete
#         from image_clusters.cluster_labels as labels
#         where labels.community_name=:community_name
#         returning *
#         ''')
#     return [dict(res) for res in connection.execute(query, community_name=community_name)]


# def get_community_names():
#     connection = get_connection(DB_CONFIG['social'])
#     query = text('select distinct cp.community_name from social.communities_posts cp')
#     return [res[0] for res in connection.execute(query)]

# CREATE TABLE IF NOT EXISTS social.posts_topics(
#     topic_id int,
#     post_id bigint)


# CREATE TABLE IF NOT EXISTS social.topics(
#     topic_id serial PRIMARY KEY,
#     type varchar(50),
#     hashtag varchar(100) UNIQUE,
#     topic varchar(100) UNIQUE)
