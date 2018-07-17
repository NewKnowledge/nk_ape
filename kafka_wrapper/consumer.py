import json
import os

from kafka import KafkaConsumer
from nk_ape import Ape

from kafka_config import KAFKA_BROKERS, KAFKA_TOPIC

ape = Ape(verbose=True)

current_path = os.path.dirname(os.path.abspath(__file__))
exclude_path = os.path.join(current_path, 'exclude_words.txt')
if os.path.isfile(exclude_path):
    with open(exclude_path) as exclude_file:
        EXCLUDE_WORDS = [word.strip('\n') for word in exclude_file.readlines()]


def main():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="kafka_consumer_group",
    )
    for message in consumer:
        if hasattr(message, 'value'):
            parsed_message = json.loads(message.value.decode("utf-8"))

            doc = parsed_message.get('content')
            post_id = parsed_message.get('post_id')
            if doc and post_id:
                # drop stop words
                doc = [d for d in doc.split(' ') if d.lower() not in EXCLUDE_WORDS]
                # run through ape
                classes = ape.get_top_classes(doc)
                print('ape consumer result:', classes)

                # write results to db

            else:
                print('parsed message missing content field')
        else:
            print('ape consumer: invalid message')


if __name__ == '__main__':
    main()
