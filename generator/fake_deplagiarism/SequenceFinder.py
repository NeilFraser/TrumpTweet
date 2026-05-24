# The code in this file was written by Gemini.


class SequenceLCSFinder:
    def __init__(self, database_sequences):
        """
        Preprocesses the database of sequences using an optimized Suffix Array.
        database_sequences: List of lists containing string or numeric tokens.
        """
        # Standardize all tokens to strings
        self.database = [[str(token) for token in seq] for seq in database_sequences]

        self.flat_list = []
        self.index_to_seq = []  # Maps flat_list index back to (sequence_index, token_index)

        # Flatten the database using unique separator tokens
        for seq_idx, seq in enumerate(self.database):
            for token_idx, token in enumerate(seq):
                self.flat_list.append(token)
                self.index_to_seq.append((seq_idx, token_idx))

            # Use a unique null-byte wrapped separator to prevent cross-sequence matches
            self.flat_list.append(f"\x00_SEP_{seq_idx}_\x00")
            self.index_to_seq.append((None, None))

        # Build the suffix array
        self.sa = self._build_suffix_array()

    def _build_suffix_array(self):
        """Builds a suffix array using the prefix-doubling algorithm."""
        n = len(self.flat_list)

        # Initial ranking based on alphabetical order of unique tokens
        unique_elements = sorted(list(set(self.flat_list)))
        rank_map = {el: i for i, el in enumerate(unique_elements)}
        ranks = [rank_map[el] for el in self.flat_list]

        sa = list(range(n))
        k = 1

        # Since your maximum sequence length is ~30, we can stop doubling
        # as soon as k exceeds our maximum window size (k <= 32).
        while k <= 32:
            pairs = [(ranks[i], ranks[i + k] if i + k < n else -1) for i in range(n)]
            sa.sort(key=lambda idx: pairs[idx])

            new_ranks = [0] * n
            new_ranks[sa[0]] = 0
            for i in range(1, n):
                prev, curr = sa[i - 1], sa[i]
                if pairs[prev] == pairs[curr]:
                    new_ranks[curr] = new_ranks[prev]
                else:
                    new_ranks[curr] = new_ranks[prev] + 1

            ranks = new_ranks
            if ranks[sa[-1]] == n - 1:
                break
            k *= 2

        return sa

    def find_longest_common_substring(self, new_sequence):
        """
        Finds the longest common substring between the new sequence and the database.
        Returns a dictionary with match metadata.
        """
        query = [str(token) for token in new_sequence]
        best_len = 0
        best_match = {
            "length": 0,
            "substring": [],
            "database_sequence_index": None,
            "database_token_index": None
        }

        # Check the longest common prefix for every suffix of your new sequence
        for start_i in range(len(query)):
            query_suff = query[start_i:]

            # Binary search to find where this query suffix would sit in the Suffix Array
            low = 0
            high = len(self.sa)
            while low < high:
                mid = (low + high) // 2
                sa_idx = self.sa[mid]

                # Calculate match length
                match_len = 0
                while (match_len < len(query_suff) and sa_idx + match_len < len(self.flat_list)):
                    if query_suff[match_len] == self.flat_list[sa_idx + match_len]:
                        match_len += 1
                    else:
                        break

                if match_len == len(query_suff):
                    high = mid
                elif sa_idx + match_len >= len(self.flat_list):
                    low = mid + 1
                else:
                    if query_suff[match_len] > self.flat_list[sa_idx + match_len]:
                        low = mid + 1
                    else:
                        high = mid

            # The closest lexicographical matches will be at 'low' or 'low - 1'
            for cand_pos in (low, low - 1):
                if 0 <= cand_pos < len(self.sa):
                    sa_idx = self.sa[cand_pos]

                    # Compute actual match length
                    match_len = 0
                    while (match_len < len(query_suff) and sa_idx + match_len < len(self.flat_list)):
                        if query_suff[match_len] == self.flat_list[sa_idx + match_len]:
                            match_len += 1
                        else:
                            break

                    # Update if we found a strictly longer common match
                    if match_len > best_len:
                        seq_idx, token_idx = self.index_to_seq[sa_idx]
                        if seq_idx is not None:  # Ensure it's not pointing to a separator
                            best_len = match_len
                            best_match = {
                                "length": best_len,
                                "substring": query[start_i : start_i + best_len],
                                "database_sequence_index": seq_idx,
                                "database_token_index": token_idx
                            }

        return best_match
