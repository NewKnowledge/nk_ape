import sys
import time
import numpy as np
from operator import itemgetter
from inflection import pluralize

from class_tree import EmbeddedClassTree, tree_score
from embedding import Embedding
from utils import (get_dropped, mean_of_rows, no_op, normalize_text,
                   unit_norm_rows)


class EmbeddedConcepts:

    def __init__(self, embedding_model, concepts,
                 max_num_samples=1e5, embed_concepts=True, verbose=False):
        self.max_num_samples = max_num_samples

        self.vprint = print if verbose else no_op

        assert isinstance(embedding_model, Embedding)
        self.embedding = embedding_model

        if embed_concepts:
            self.embed_concepts(concepts)

    def format_concepts(self, concepts):
        # list of lists of single words
        word_groups = np.array([normalize_text(text) for text in concepts])
        return self.embedding.remove_out_of_vocab(word_groups)

    def embed_concepts(self, concepts=None):
        self.vprint('embedding concepts')

        concept_targets = concepts if concepts else self.concepts
        concept_targets = self.format_concepts(concept_targets)
        # compute data embedding for the target concepts
        self.vprint('computing word embedding')
        try:
            if self.max_num_samples and \
                    len(concept_targets) > self.max_num_samples:
                self.vprint(
                    'subsampling rows from length {0} to {1}'
                    .format(len(concept_targets), self.max_num_samples))
                np.random.shuffle(concept_targets)  # TODO minibatches?
                concept_targets = concept_targets[:self.max_num_samples]

            # matrix of w/ len(concept_targets) rows and n_emb_dim columns
            dat_vecs = np.array(
                [self.embedding.embed_multi_words(words)
                    for words in concept_targets])
            self.concept_vectors = unit_norm_rows(dat_vecs)
        except:
            print(sys.exc_info())


class ConceptDescriptor:

    def __init__(self,
                 concepts,
                 tree='ontologies/class-tree_dbpedia_2016-10.json',
                 embedding='models/wiki2vec/en.model',
                 row_agg_func=mean_of_rows,
                 tree_agg_func=np.mean,
                 max_num_samples=1e6,
                 verbose=False
                 ):

        # print function that works only when verbose is true
        self.vprint = print if verbose else no_op
        self.max_num_samples = max_num_samples

        # load embeddings, concept vecs, and KB
        self.embedding = embedding if isinstance(embedding, Embedding) else \
            Embedding(embedding_path=embedding, verbose=verbose)
        self.concepts = concepts if isinstance(concepts, EmbeddedConcepts) \
            else EmbeddedConcepts(self.embedding, concepts, max_num_samples,
                                  verbose=verbose)
        self.tree = tree if isinstance(tree, EmbeddedClassTree) else \
            EmbeddedClassTree(self.embedding, tree_path=tree, verbose=verbose)

        self.row_agg_func = row_agg_func
        self.tree_agg_func = tree_agg_func

        self.similarity_matrix = {}

    @property
    def classes(self):
        return self.tree.classes

    def compute_similarity_matrix(self):

        class_matrix = self.tree.class_vectors.T

        # compute cosine similarity bt embedded data and ontology classes
        self.vprint('computing class similarity for target concepts')

        sim_mat = np.dot(self.concepts.concept_vectors, class_matrix)
        self.similarity_matrix = sim_mat

    def get_concept_class_scores(self):

        if not self.similarity_matrix:
            self.vprint('computing similarity matrix')
            self.compute_similarity_matrix()

        self.vprint('aggregating row scores')
        sim_scores = self.row_agg_func(self.similarity_matrix)

        self.vprint('aggregating tree scores')
        return self.aggregate_tree_scores(sim_scores)  # same

    def get_concept_description(self):
        final_scores = self.get_concept_class_scores()
        top_word = self.tree.classes[np.argmax(final_scores)]
        description = (
            'These concepts can be summarized as {0}.'
            .format(pluralize(top_word)))
        self.vprint('\n\n concept set description:', description, '\n\n')

        return(description)

    def get_top_n_words(self, n):
        final_scores = self.get_concept_class_scores()
        indexed_scores = zip(final_scores, range(len(final_scores)))
        indexed_scores = sorted(
            indexed_scores, key=itemgetter(0), reverse=True)
        top_n = indexed_scores[0:n]
        top_words = [{'concept': self.tree.classes[index], 'conf': score} for (score, index) in top_n]

        return top_words

    def aggregate_tree_scores(self, scores):
        # convert score to dict that maps class to score if needed
        score_map = (scores if isinstance(scores, dict) else
                     dict(zip(self.tree.classes, scores)))

        # aggregate score over tree structure
        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)

        # convert returned score map back to array
        return np.array([agg_score_map[cl] for cl in self.tree.classes])


class Ape:
    ''' ApeListener takes a string of space-separated concepts
        and produces a set of related and (possibly) more abstract
        concepts as JSON output
    '''

    def __init__(self, embedding_path):
        self.tree = 'ontologies/class-tree_dbpedia_2016-10.json'
        self.embedding = embedding_path
        self.row_agg_func = mean_of_rows
        self.tree_agg_func = np.mean
        self.source_agg_func = mean_of_rows
        self.max_num_samples = 1e6
        self.n_words = 10
        self.verbose = True

    def predict_labels(self, concept_string):

        if not isinstance(concept_string, (list, tuple)):
            concept_string = concept_string.split(' ')

        start = time.time()

        ape = ConceptDescriptor(
            concepts=concept_string,
            tree=self.tree,
            embedding=self.embedding,
            row_agg_func=self.row_agg_func,
            tree_agg_func=self.tree_agg_func,
            max_num_samples=self.max_num_samples,
            verbose=self.verbose)

        print(
            "Ape took %f seconds to execute"
            % (time.time()-start))

        return ape.get_top_n_words(self.n_words)


if __name__ == '__main__':
    client = Ape(embedding_path='/Users/ghonk/nk_projects/ape/embeddings/wiki2vec/en.model')
    test_concepts = ['gorilla', 'chimp', 'orangutan', 'gibbon', 'human']
    result = client.predict_labels(test_concepts)
    print(result)
