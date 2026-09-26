import pandas as pd

DATA_DIR = "../data"
COLLECTION = DATA_DIR + "/collection.csv"
TOPICS = DATA_DIR + "/topics.csv"
GROUNDTRUTH = DATA_DIR + "/groundtruth.csv"


def create_qrels(topics_filename, groundtruth_filename):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)
    data = pd.merge(topics, groundtruth, on='topic_id')
    data['subtopic'] = 0
    print(data.shape)
    qrels = data[['question_id', 'subtopic', 'passage_id', 'relevance_judgment']]
    print(qrels.head())
    qrels.to_csv(DATA_DIR + "/qrels.txt", sep='\t', index=False, header=False)


def create_pyserini_collection(collection_filename):
    collection = pd.read_csv(collection_filename)
    collection.columns = ['id', 'contents']
    collection.to_json(DATA_DIR + "/collection.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    create_qrels(TOPICS, GROUNDTRUTH)
    create_pyserini_collection(COLLECTION)
