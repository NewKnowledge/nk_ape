import json
import logging
import os

from kafka import KafkaConsumer
from nk_ape import Ape

from kafka_config import KAFKA_BROKERS, KAFKA_TOPIC
from queries import insert_classes

logging.basicConfig(level=logging.INFO)

print('loading ape')
ape = Ape(verbose=True)
print('finished loading ape')

current_path = os.path.dirname(os.path.abspath(__file__))
exclude_path = os.path.join(current_path, 'exclude_words.txt')
if os.path.isfile(exclude_path):
    with open(exclude_path) as exclude_file:
        EXCLUDE_WORDS = [word.strip('\n') for word in exclude_file.readlines()]


def main():
    print('creating kafka consumer')
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="kafka_consumer_group",
    )
    for message in consumer:
        print(f'message being consumed:{message}')
        try:
            if hasattr(message, 'value'):
                parsed_message = json.loads(message.value.decode("utf-8"))

                doc = parsed_message.get('content')
                content_id = parsed_message.get('content_id')
                print(f'doc and content id {doc}, {content_id}')
                if doc and content_id:
                    # drop stop words
                    doc = [d for d in doc.split(' ') if d.lower() not in EXCLUDE_WORDS]
                    # run through ape
                    scored_classes = ape.get_top_classes(doc)
                    print(f'ape consumer result: {scored_classes}')
                    # write results to db
                    insert_classes(content_id, scored_classes)
                    print(f'results inserted: {scored_classes}')
                else:
                    print('parsed message missing content field or content_id')
            else:
                print('message missing value attribute')
        except Exception as err:
            logging.error(err)
            logging.error(f'failed to consume message {message}')


if __name__ == '__main__':
    main()
