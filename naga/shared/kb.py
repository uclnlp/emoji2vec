# coding=utf-8
# Knowledge Base Representation
# Tim Rocktäschel, Guillaume Bouchard

import random
import numpy as np
import pandas as pd


TRAIN_LABEL = "train"
DEV_LABEL = "dev"
TEST_LABEL = "test"
TMP_LABEL = "tmp"


class KB:
    """
     KB represents a knowledge base of facts of varying arity
     >>> kb = KB()
     >>> kb.add_train("r1", "e1", "e2")
     >>> kb.is_true("r1", "e1", "e2")
     True

     Anything can be used to represent symbols
     >>> kb.add_train("r2", ("e1", "e3"))
     >>> kb.is_true("r2", ("e1", "e3"))
     True

     >>> kb.add_train("r2", "e1", "e3")
     >>> kb.add_train("r3", "e1", "e2", "e3")
     >>> kb.add_train("r4", ("e4", "e5"), "e6")
     >>> kb.add_train("r5", "e4")

     Any fact can be queried
     >>> kb.is_true("r1", "e1", "e2", "e4", "e5", "e6")
     False

     >>> kb.get_facts("e1", 1)
     [(('r1', 'e1', 'e2'), True, 'train'), (('r2', 'e1', 'e3'), True, 'train'), (('r3', 'e1', 'e2', 'e3'), True, 'train')]

     Adding the same fact twice does not add it
     >>> kb.add_train("r1", "e1", "e2")
     >>> len(kb.get_facts("e1", 1)) == 3
     True

     >>> kb.get_facts("unk_rel", 1)
     []
     >>> kb.get_facts("unk_ent", 6)
     []

     >>> kb.dim_size(0)
     5
     >>> kb.dim_size(1)
     4
     >>> kb.dim_size(3)
     1

    >>> [x for x in kb.get_all_facts_of_arity(1)]
    [(('r2', ('e1', 'e3')), True, 'train'), (('r5', 'e4'), True, 'train')]
    >>> [x for x in kb.get_all_facts_of_arity(2)]
    [(('r1', 'e1', 'e2'), True, 'train'), (('r2', 'e1', 'e3'), True, 'train'), (('r4', ('e4', 'e5'), 'e6'), True, 'train')]

     >>> sorted(list(kb.get_symbols(0)))
     ['r1', 'r2', 'r3', 'r4', 'r5']
     >>> sorted(list(kb.get_symbols(2)))
     ['e2', 'e3', 'e6']

     >>> kb.get_vocab(0)
     ['r1', 'r2', 'r3', 'r4', 'r5']
     >>> kb.get_vocab(1)
     ['e1', ('e1', 'e3'), ('e4', 'e5'), 'e4']
     >>> kb.get_vocab(2)
     ['e2', 'e3', 'e6']

     >>> np.random.seed(0)
     >>> kb.sample_neg("r1", 0, 2) not in kb.get_all_facts()
     True
     """

    def __init__(self):
        #random.seed(0)
        # holds all known facts for every arity
        self.__facts = {}
        # holds all facts independent of arity
        self.__all_facts = set()
        # holds set of all symbols in every dimension
        self.__symbols = list()
        # holds list of all symbols in every dimension
        self.__vocab = list()
        # holds mappings of symbols to indices in every dimension
        self.__ids = list()
        # holds known facts for symbols in every dimension
        self.__maps = list()
        # caches number of dimensions since len(...) is slow
        self.__dims = list()
        # global mapping of symbols to indices independent from dimension
        self.__global_ids = {}
        self.__formulae = {}

    def __add_to_facts(self, fact):
        arity = len(fact[0]) - 1
        if arity not in self.__facts:
            self.__facts[arity] = list()
        self.__facts[arity].append(fact)
        self.__all_facts.add(fact)

    def __remove_from_facts(self, fact):
        arity = len(fact[0]) - 1
        if arity in self.__facts:
            self.__facts[arity].remove(fact)
            self.__all_facts.remove(fact)
        # todo: necessary?
        self.__maps[0][fact[0][0]].remove(fact)

    def __add_word(self, word):
        if word not in self.__global_ids:
            self.__global_ids[word] = len(self.__global_ids)

    def __add_to_symbols(self, key, dim):
        if len(self.__symbols) <= dim:
            self.__symbols.append(set())
        self.__symbols[dim].add(key)

        words = key
        if isinstance(words, str):
            words = [key]
        for word in words:
            self.__add_word(word)

    def __add_to_vocab(self, key, dim):
        if len(self.__vocab) <= dim:
            self.__vocab.append(list())
            self.__ids.append({})
            self.__dims.append(0)
        if len(self.__symbols) <= dim or key not in self.__symbols[dim]:
            self.__ids[dim][key] = len(self.__vocab[dim])
            self.__vocab[dim].append(key)
            self.__dims[dim] += 1

    def __add_to_maps(self, key, dim, fact):
        if len(self.__maps) <= dim:
            self.__maps.append({key: list()})
        if key in self.__maps[dim]:
            self.__maps[dim][key].append(fact)
        else:
            self.__maps[dim].update({key: [fact]})

    def get_all_facts_of_arity(self, arity, typ=TRAIN_LABEL):
        if arity not in self.__facts:
            return set()
        else:
            return filter(lambda x: x[2] == typ, self.__facts[arity])

    def get_all_facts(self, of_types=set()):
        if len(of_types) == 0:
            return self.__all_facts
        else:
            return [x for x in self.__all_facts if x[2] in of_types]

    def add(self, truth, typ, *keys):
        assert isinstance(truth, bool)
        if not self.contains_fact(truth, typ, *keys):
            fact = (keys, truth, typ)
            self.__add_to_facts(fact)
            for dim in range(len(keys)):
                key = keys[dim]
                self.__add_to_vocab(key, dim)
                self.__add_to_symbols(key, dim)
                self.__add_to_maps(key, dim, fact)

    def contains_fact(self, truth, typ, *keys):
        return (keys, truth, typ) in self.get_all_facts()

    def contains_fact_any_type(self, truth, *keys):
        return self.contains_fact(truth, TRAIN_LABEL, *keys) or \
               self.contains_fact(truth, DEV_LABEL, *keys) or \
               self.contains_fact(truth, TEST_LABEL, *keys) or \
               self.contains_fact(truth, TMP_LABEL, *keys)

    def add_train(self, *keys):
        self.add(True, TRAIN_LABEL, *keys)

    def get_facts(self, key, dim, typ=None):
        result = list()
        if len(self.__maps) > dim:
            if key in self.__maps[dim]:
                result = self.__maps[dim][key]
        if typ is None:
            return result
        else:
            return [f for f in result if f[2] == typ]

    def is_true(self, *keys):
        arity = len(keys) - 1
        if arity not in self.__facts:
            return False
        else:
            return (keys, True, TRAIN_LABEL) in self.__facts[arity]

    def dim_size(self, dim):
        if dim >= len(self.__dims):
            return 0
        else:
            return self.__dims[dim]

    # @profile
    def sample_neg(self, key, dim, arity, oracle=False, tries=100):
        cell = list()
        for i in range(0, arity + 1):
            symbol_ix = np.random.randint(0, self.dim_size(i))
            symbol = self.__vocab[i][symbol_ix]
            cell.append(symbol)
        cell[dim] = key
        cell = tuple(cell)

        if tries == 0:
            print("Warning, couldn't sample negative fact for",
                  key, "in dim", dim)
            return cell, False, TRAIN_LABEL
        elif (cell, True, TRAIN_LABEL) in self.__facts[arity] or \
                (oracle and self.contains_fact_any_type(True, *cell)):
            return self.sample_neg(key, dim, arity, oracle, tries - 1)
        else:
            return cell, False, TRAIN_LABEL

    def get_vocab(self, dim):
        return self.__vocab[dim]

    def get_symbols(self, dim):
        return self.__symbols[dim]

    def get_global_vocab(self):
        return self.__global_ids

    def to_data_frame(self):
        data = {}
        for key1 in self.__vocab[0]:
            row = list()
            for key2 in self.__vocab[1]:
                if ((key1, key2), True, TRAIN_LABEL) in self.__facts[1]:
                    row.append(1.0)
                elif ((key1, key2), False, TRAIN_LABEL) in self.__facts[1]:
                    row.append(0.0)
                elif ((key1, key2), True, TEST_LABEL) in self.__facts[1]:
                    row.append("*1")
                elif ((key1, key2), False, TEST_LABEL) in self.__facts[1]:
                    row.append("*0")
                elif ((key1, key2), True, TMP_LABEL) in self.__facts[1]:
                    row.append("**1")
                elif ((key1, key2), True, TMP_LABEL) in self.__facts[1]:
                    row.append("**0")
                else:
                    row.append("")
            data[key1] = row
        # for some reason, the columns aren't printed in the correct order,
        # so they must be manually set
        df = pd.DataFrame(data, index=self.__vocab[1], columns=self.__vocab[0])
        return df

    def get_id(self, key, dim):
        return self.__ids[dim][key]

    def get_ids(self, *keys):
        ids = list()
        for dim in range(len(keys)):
            ids.append(self.get_id(keys[dim], dim))
        return ids

    def get_global_id(self, symbol):
        return self.__global_ids[symbol]

    def get_global_ids(self, *symbols):
        ids = list()
        for symbol in symbols:
            # fixme
            if not isinstance(symbol, str):
                for s in symbol:
                    ids.append(self.get_global_id(s))
            else:
                ids.append(self.get_global_id(symbol))
        return ids

    def num_global_ids(self):
        return len(self.__global_ids)

    def get_key(self, id, dim):
        return self.__vocab[dim][id]

    def get_keys(self, *ids):
        keys = list()
        for dim in range(len(ids)):
            keys.append(self.get_key(ids[dim], dim))
        return keys

    def add_formulae(self, label, formulae):
        self.__formulae[label] = formulae

    def get_formulae(self, label):
        if label in self.__formulae:
            return self.__formulae[label]
        else:
            return []

    def get_formulae_for_ntp(self):
        result = []
        for label in self.__formulae:
            formulae = self.__formulae[label]

            if label == "inv":
                for (lhs, rhs) in formulae:
                    result.append(([[lhs, "X", "Y"]], [rhs, "Y", "X"]))
            elif label == "impl":
                for arity in formulae:
                    for (lhs, rhs) in formulae[arity]:
                        if arity == 2:
                            result.append(([[lhs, "X", "Y"]], [rhs, "X", "Y"]))
                        else:
                            #todo
                            pass
            elif label == "impl_conj":
                for arity in formulae:
                    for (lhs1, lhs2, rhs) in formulae[arity]:
                        if arity == 2:
                            result.append(([[lhs1, "X", "Y"], [lhs2, "X", "Y"]],
                                           [rhs, "X", "Y"]))
                        else:
                            #todo
                            pass
            elif label == "trans":
                for (lhs1, lhs2, rhs) in formulae:
                    result.append(([[lhs1, "X", "Y"], [lhs2, "Y", "Z"]],
                                   [rhs, "X", "Z"]))
        return result

    def get_formulae_strings(self):
        result = []
        for label in self.__formulae:
            formulae = self.__formulae[label]
            if isinstance(formulae, list):
                # todo
                pass
            else:
                for arity in formulae:
                    for formula in formulae[arity]:
                        if label == "impl":
                            (body, head) = formula
                            result.append(body + " => " + head)
                        elif label == "impl_conj":
                            (arg1, arg2, head) = formula
                            result.append(arg1 + " ^ " + arg2 + " => " + head)
                        else:
                            # todo
                            pass
        return result

    def add_train_stochastic(self, test_prob, *keys):
        if np.random.uniform(0, 1.0) >= test_prob:
            self.add(True, TRAIN_LABEL, *keys)
        else:
            self.add(True, TEST_LABEL, *keys)

    @staticmethod
    def id_args(args):
        return args

    @staticmethod
    def inv_args(args):
        return args[::-1]

    def add_tmp_inferred_fact(self, facts, head, arg_fun):
        done = True
        for ((rel, args), target, typ) in facts:
            contained = self.contains_fact_any_type(target, head, arg_fun(args))
            if target and not contained:
                self.add(True, TMP_LABEL, head, arg_fun(args))
                done = False
        return done

    def apply_formulae(self, test_prob=0.0, sampled_unobserved_per_true=1):
        done = False
        while not done:
            done = True

            for (body, head) in self.get_formulae("inv"):
                facts = self.get_facts(body, 0)
                done = self.add_tmp_inferred_fact(facts, head, self.inv_args)

            for arity in range(1, 4):
                if arity in self.get_formulae("impl"):
                    for (body, head) in self.get_formulae("impl")[arity]:
                        facts = self.get_facts(body, 0)
                        done = self.add_tmp_inferred_fact(facts, head,
                                                          self.id_args)

                if arity in self.get_formulae("impl_conj"):
                    for (body1, body2, head) in \
                            self.get_formulae("impl_conj")[arity]:
                        facts1 = self.get_facts(body1, 0)
                        facts2 = self.get_facts(body2, 0)
                        facts = [x for x in facts1 for y in facts2
                                 if x[0][1] == y[0][1] and
                                 x[1] == y[1] and x[2] == y[2]]
                        done = self.add_tmp_inferred_fact(facts, head,
                                                          self.id_args)

            for (body1, body2, head) in self.get_formulae("trans"):
                facts1 = self.get_facts(body1, 0)
                facts2 = self.get_facts(body2, 0)
                facts = [((head, (e1, e4)), typ1, target1) for
                         ((rel1, (e1, e2)), typ1, target1) in facts1 for
                         ((rel2, (e3, e4)), typ2, target2) in facts2
                         if e2 == e3 and typ1 == typ2 and target1 == target2]
                done = self.add_tmp_inferred_fact(facts, head, self.id_args)

        # print(self.to_data_frame())

        # map tmp facts to test or train based on test_prob
        rels = list(self.get_symbols(0))
        # to make sure we sample the same test facts when given a random seed
        rels.sort()
        for rel in rels:
            facts = [fact for fact in self.get_facts(rel, 0)
                     if fact[2] == TMP_LABEL]
            for (key, truth, typ) in facts:
                self.__remove_from_facts((key, truth, typ))
                new_typ = TRAIN_LABEL
                if np.random.uniform(0, 1) < test_prob:
                    new_typ = TEST_LABEL
                self.add(truth, new_typ, *key)

        # sample unobserved facts as negative test facts
        for rel in self.get_symbols(0):
            test_facts = [fact for fact in self.get_facts(rel, 0)
                          if fact[2] == TEST_LABEL]
            #print(rel, test_facts)
            for test_fact in test_facts:
                #print(test_fact)
                for i in range(0, sampled_unobserved_per_true):
                    (key, truth, typ) = self.sample_neg(rel, 0, 1, oracle=True)
                    self.add(False, TEST_LABEL, *key)


# implements an iterator over training examples while sampling negative examples
class BatchNegSampler:
    # todo: pass sampling function as argument
    def __init__(self, kb, arity, batch_size, neg_per_pos):
        self.kb = kb
        self.batch_size = batch_size
        self.facts = list(self.kb.get_all_facts_of_arity(arity))
        self.todo_facts = list(self.facts)
        self.num_facts = len(self.facts)
        self.neg_per_pos = neg_per_pos
        self.__reset()

    # @profile
    def __reset(self):
        self.todo_facts = list(self.facts)
        random.shuffle(self.todo_facts)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.num_facts:
            self.__reset()
            raise StopIteration
        return self.get_batch(neg_per_pos=self.neg_per_pos)

    # todo: generalize this towards sampling in different dimensions
    # @profile
    def get_batch(self, neg_per_pos=1):
        if self.count >= self.num_facts:
            self.__reset()
        num_pos = int(self.batch_size / (1 + neg_per_pos))
        pos = self.todo_facts[0:num_pos]
        self.count += self.batch_size
        self.todo_facts = self.todo_facts[num_pos::]
        neg = list()
        for fact in pos:
            for i in range(neg_per_pos):
                neg.append(self.kb.sample_neg(fact[0][0], 0, 1))
        return self.__tensorize(pos + neg)

    # @profile
    def __tensorize(self, batch):
        rows = list()
        cols = list()
        targets = list()

        for i in range(len(batch)):
            example = batch[i]
            rows.append(self.kb.get_id(example[0][0], 0))
            cols.append(self.kb.get_id(example[0][1], 1))
            if example[1]:
                targets.append(1)
            else:
                targets.append(0)

        return rows, cols, targets

    def get_epoch(self):
        return self.count / float(self.num_facts)


class Seq2Fact2SeqBatchSampler:
    # todo: pass sampling function as argument
    def __init__(self, kb, arity, batch_size):
        self.kb = kb
        self.batch_size = batch_size
        self.facts = filter(lambda x: not x[0][0][0].startswith("REL$"), list(self.kb.get_all_facts_of_arity(arity)))
        self.todo_facts = list(self.facts)
        self.num_facts = len(self.facts)
        self.__reset()

    # @profile
    def __reset(self):
        self.todo_facts = list(self.facts)
        random.shuffle(self.todo_facts)
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.num_facts:
            self.__reset()
            raise StopIteration
        return self.get_batch()

    # todo: generalize this towards sampling in different dimensions
    # @profile
    def get_batch(self):
        if self.count >= self.num_facts:
            self.__reset()
        pos = self.todo_facts[0:self.batch_size]
        self.count += self.batch_size
        self.todo_facts = self.todo_facts[self.batch_size::]
        return self.__tensorize(pos)

    # @profile
    def __tensorize(self, batch):
        rows = list()
        cols = list()

        for i in range(len(batch)):
            example = batch[i]
            rows.append(self.kb.get_global_ids(example[0][0]))
            cols.append(self.kb.get_id(example[0][1], 1))

        return rows, cols, rows

    def get_epoch(self):
        return self.count / float(self.num_facts)


def test_pairs():
    kb = KB()
    kb.add_train("r4", ("e4", "e5"), "e6")
    kb.add_train("r1", ("e4", "e5"), "e6")
    kb.add_train("r2", ("e4", "e5"), "e6")
    print(kb.get_facts(("e4", "e5"), 1))


def test_two_kbs():
    kb1 = KB()
    kb1.add_train("blah", "keks")
    kb2 = KB()
    kb2.add_train("blubs", "hui")
    print(kb2.get_all_facts())


def test_global_ids():
    kb = KB()
    kb.add_train("blah", "keks")
    kb.add_train(("e1", "e2"), "r")
    print(kb.get_global_id("keks"))
    print(kb.get_global_id("e1"))
    print(kb.get_global_id("r"))
    print(kb.get_global_ids("e1", "e2", "r"))

    # test_kb()
    # test_sampling()
    # test_train_cells()
    # test_pairs()
    # test_two_kbs()
    # test_global_ids()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
