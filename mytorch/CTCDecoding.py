import numpy as np
from itertools import groupby

def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        print("symbol_set: {}".format(self.symbol_set))
        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        for i in range(len(y_probs[0])): # over seq length
            # for j in range(len(y_probs)):
            probs = y_probs[:,i,0]
            # print(type(probs))
            # break
            path_prob *= np.max(probs)
            sym_ind = np.argmax(probs)
            if sym_ind == 0:
                decoded_path.append(" ")
            else:
                decoded_path.append(self.symbol_set[sym_ind-1])



            # break
        decoded_path = clean_path(decoded_path)
        decoded_path = list(decoded_path)
        """ removing consecutive repetitions """
        decoded_path = [i[0] for i in groupby(decoded_path)]
        decoded_path = clean_path(decoded_path)
        """ removing spaces """
        decoded_path = decoded_path.split(" ")
        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob
        # raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def InitializePaths(self, symbol_set, y):

        InitialBlankPathScore = {}
        InitialPathScore = {}
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ""
        InitialBlankPathScore[path] = y[0]  # Score of blank at t=1
        InitialPathsWithFinalBlank = set()
        InitialPathsWithFinalBlank.add(path)
        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = set()
        for c in range(len(symbol_set)):  # This is the entire symbol set, without the blank
            path = symbol_set[c]
            InitialPathsWithFinalSymbol.add(path)
            InitialPathScore[path] = y[c+1]

        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = {}
        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

        for path in PathsWithTerminalSymbol:
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0]

            else:
                UpdatedPathsWithTerminalBlank.add(path)
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):

        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}
        for path in PathsWithTerminalBlank:
            for c in range(len(self.symbol_set)):
                newpath = path + self.symbol_set[c]
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = BlankPathScore[path] * y[c + 1]

        for path in PathsWithTerminalSymbol:
            for c in range(len(self.symbol_set)):
                newpath = path if (self.symbol_set[c] == path[-1]) else path + self.symbol_set[c]
                if (newpath in UpdatedPathsWithTerminalSymbol):
                    UpdatedPathScore[newpath] += PathScore[path] * y[c + 1]
                else:
                    UpdatedPathsWithTerminalSymbol.add(newpath)
                    UpdatedPathScore[newpath] = PathScore[path] * y[c + 1]
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        scores = []
        for i in PathsWithTerminalBlank:
            scores.append(BlankPathScore[i])
        for i in PathsWithTerminalSymbol:
            scores.append(PathScore[i])

        scores = sorted(scores, reverse=True)

        if (self.beam_width < len(scores)):
            cutoff = scores[self.beam_width]
        else:
            cutoff = scores[-1]

        PrunedPathsWithTernimalBlank = set()
        for i in PathsWithTerminalBlank:
            if (BlankPathScore[i] > cutoff):
                PrunedPathsWithTernimalBlank.add(i)
                PrunedBlankPathScore[i] = BlankPathScore[i]

        PrunedPathsWithTernimalSymbol = set()
        for i in PathsWithTerminalSymbol:
            if (PathScore[i] > cutoff):
                PrunedPathsWithTernimalSymbol.add(i)
                PrunedPathScore[i] = PathScore[i]

        return PrunedPathsWithTernimalBlank, PrunedPathsWithTernimalSymbol, PrunedBlankPathScore, PrunedPathScore

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore_id, PathsWithTerminalSymbol, PathScore_id):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore_id
        # Paths with terminal blanks will contribute scores to existing identical paths from # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore_id[p]
            else:
                MergedPaths.add(p)  # Set addition
                FinalPathScore[p] = BlankPathScore_id[p]

        return MergedPaths, FinalPathScore


    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores


        """
        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None

        best_path, merged_path_scores = None, None

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return      

        path_score = []
        blank_path_score = []

        """ First time instant: Initialize paths with each of the symbols,
            including blank, using score at time t=1 """
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(self.symbol_set, y_probs[:, 0, 0])

        """ Subsequent time steps """
        for t in range(1, len(y_probs[0])):

            # Prune the collection down to the BeamWidth
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                  NewBlankPathScore, NewPathScore)
            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank,
                                                                                PathsWithTerminalSymbol,
                                                                                y_probs[:, t, 0],
                                                                                BlankPathScore,
                                                                                PathScore
                                                                                )
            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(
                                                                            PathsWithTerminalBlank,
                                                                            PathsWithTerminalSymbol,
                                                                            y_probs[:, t, 0],
                                                                            BlankPathScore, PathScore
                                                                            )

        """ Merge identical paths differing only by the final blank """
        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore,
                                                          NewPathsWithTerminalSymbol, NewPathScore)

        """Pick best path"""
        best_path = max(FinalPathScore, key=FinalPathScore.get)  # Find the path with the best score


        return best_path, FinalPathScore
        # raise NotImplementedError

